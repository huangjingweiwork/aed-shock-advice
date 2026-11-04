"""The compression tools: structured pruning, unstructured pruning, and int8 quantisation via TensorFlow Lite."""
import gzip, time
import numpy as np, tensorflow as tf
from tensorflow import keras
from models import large


# ---------------------------------------------------------------- structured pruning (whole filters removed)

def l1(kernel):
    """L1 norm (sum of |weights|) of each output filter, divided by the layer mean so layers can be added."""
    s = np.abs(kernel).reshape(-1, kernel.shape[-1]).sum(0)
    return s / s.mean()


def top(score, k):
    """Indices of the k highest scores, in their original order."""
    return np.sort(np.argsort(score)[::-1][:k])


def prune_structured(model, keep):
    """Keep the highest-L1 filters in every layer and copy their weights into a narrower copy of the network.

    Removing filter j of a layer also removes input channel j of the next layer. Channels that meet at a
    residual Add must be chosen together, so both branches of an Add use one shared index set.
    """
    narrow = large(model.input.shape[1], keep)
    W = {l.name: l.get_weights() for l in model.layers if l.get_weights()}
    width = lambda name: narrow.get_layer(name).get_weights()[0].shape[-1]
    new = {}
    # Block 1 has no shortcut conv: the stem c0 is added straight to b1b, so they share one index set.
    out_idx = top(l1(W["c0"][0]) + l1(W["b1b"][0]), width("c0"))
    new["c0"] = [W["c0"][0][:, :, out_idx], W["c0"][1][out_idx]]
    in_idx = out_idx
    for i in range(1, 5):
        a = top(l1(W[f"b{i}a"][0]), width(f"b{i}a"))
        new[f"b{i}a"] = [W[f"b{i}a"][0][:, in_idx][:, :, a], W[f"b{i}a"][1][a]]
        new[f"b{i}bn"] = [p[a] for p in W[f"b{i}bn"]]
        if i > 1:  # blocks 2-4: 1x1 shortcut conv and b{i}b meet at the Add
            out_idx = top(l1(W[f"b{i}b"][0]) + l1(W[f"b{i}s"][0]), width(f"b{i}b"))
            new[f"b{i}s"] = [W[f"b{i}s"][0][:, in_idx][:, :, out_idx], W[f"b{i}s"][1][out_idx]]
        new[f"b{i}b"] = [W[f"b{i}b"][0][:, a][:, :, out_idx], W[f"b{i}b"][1][out_idx]]
        in_idx = out_idx
    d = top(l1(W["d1"][0]), width("d1"))
    new["d1"] = [W["d1"][0][in_idx][:, d], W["d1"][1][d]]
    new["out"] = [W["out"][0][d], W["out"][1]]
    for name, ws in new.items():
        narrow.get_layer(name).set_weights(ws)
    return narrow


# ---------------------------------------------------------------- unstructured pruning (single weights zeroed)

def prune_unstructured(model, sparsity):
    """Zero the smallest-magnitude `sparsity` share of the kernel weights in every conv / dense layer.

    Returns a copy of the model and the masks (1 = keep, 0 = pruned). The shape of the network is unchanged.
    """
    pruned = keras.models.clone_model(model)
    pruned.set_weights(model.get_weights())
    masks = {}
    for l in pruned.layers:
        if hasattr(l, "kernel") and l.kernel.shape[-1] > 1:  # skip the single-output layer
            k = l.kernel.numpy()
            m = (np.abs(k) > np.quantile(np.abs(k), sparsity)).astype(np.float32)
            l.kernel.assign(k * m)
            masks[l.name] = m
    return pruned, masks


class KeepPruned(keras.callbacks.Callback):
    """During fine-tuning, re-apply the masks after every batch so pruned weights stay at zero."""

    def __init__(self, masks):
        super().__init__()
        self.masks = {n: tf.constant(m) for n, m in masks.items()}

    def on_train_batch_end(self, batch, logs=None):
        for n, m in self.masks.items():
            k = self.model.get_layer(n).kernel
            k.assign(k * m)


def sparsity_of(model):
    ks = [l.kernel.numpy() for l in model.layers if hasattr(l, "kernel")]
    return float(sum((k == 0).sum() for k in ks) / sum(k.size for k in ks))


# ---------------------------------------------------------------- quantisation (TensorFlow Lite converter)

def to_tflite_fp32(model):
    """Plain conversion, no compression: the float32 reference file."""
    return tf.lite.TFLiteConverter.from_keras_model(model).convert()


def to_tflite_dynamic(model):
    """Dynamic-range quantisation: weights stored as int8, activations stay float32. Needs no data."""
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    return conv.convert()


def to_tflite_int8(model, calib):
    """Full-integer quantisation: int8 weights (one scale per output channel) and int8 activations.

    The converter runs the calibration clips through the model to record each activation's range, which sets
    that tensor's scale and zero point. Input and output are int8 too, so the chip never touches floating point.
    """
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = lambda: ([c[None]] for c in calib)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = conv.inference_output_type = tf.int8
    return conv.convert()


def run_tflite(blob, X, batch=256):
    """P(shockable) for every clip in X, computed by the TFLite interpreter exactly as stored in the file."""
    it = tf.lite.Interpreter(model_content=blob)
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    out = np.empty(len(X), np.float32)
    for s in range(0, len(X), batch):
        xb = X[s:s + batch]
        if i["dtype"] == np.int8:  # float mV -> int8 using the input tensor's scale and zero point
            sc, zp = i["quantization"]
            xb = np.clip(np.round(xb / sc + zp), -128, 127).astype(np.int8)
        it.resize_tensor_input(i["index"], xb.shape)
        it.allocate_tensors()
        it.set_tensor(i["index"], xb)
        it.invoke()
        y = it.get_tensor(o["index"]).ravel()
        if o["dtype"] == np.int8:
            sc, zp = o["quantization"]
            y = (y.astype(np.float32) - zp) * sc
        out[s:s + len(xb)] = y
    return out


def latency_ms(blob, x, runs=200):
    """Median time for one clip on this computer's CPU, single thread (not the microcontroller)."""
    it = tf.lite.Interpreter(model_content=blob, num_threads=1)
    it.allocate_tensors()
    i = it.get_input_details()[0]
    q = np.zeros(i["shape"], i["dtype"]) if i["dtype"] == np.int8 else x[None].astype(np.float32)
    it.set_tensor(i["index"], q)
    for _ in range(20):
        it.invoke()
    t = []
    for _ in range(runs):
        t0 = time.perf_counter()
        it.invoke()
        t.append(time.perf_counter() - t0)
    return 1000 * float(np.median(t))


def peak_activation_bytes(blob):
    """Largest total size of the activation tensors alive at the same moment while the model runs.

    This is a lower bound on the RAM the model needs on the chip (the runtime adds some overhead on top).
    """
    it = tf.lite.Interpreter(model_content=blob)
    ops = it._get_ops_details()
    t = {d["index"]: d for d in it.get_tensor_details()}
    first = {it.get_input_details()[0]["index"]: -1}
    for k, op in enumerate(ops):
        for i in op["outputs"]:
            first[i] = k
    last = {i: (len(ops) if i == it.get_output_details()[0]["index"] else s) for i, s in first.items()}
    for k, op in enumerate(ops):
        for i in op["inputs"]:
            if i in last:
                last[i] = max(last[i], k)
    size = lambda i: int(np.prod(t[i]["shape"])) * np.dtype(t[i]["dtype"]).itemsize
    return max(sum(size(i) for i in first if first[i] <= k <= last[i]) for k in range(len(ops)))


def gzip_bytes(blob):
    """File size after gzip: zeros left by unstructured pruning compress well, but only on disk."""
    return len(gzip.compress(blob, 9))
