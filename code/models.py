"""The two network designs. Both read one 5 s clip (625 samples) and output P(shockable)."""
from tensorflow import keras
from tensorflow.keras import layers


def small(n=625):
    """Three conv layers, 4,625 weights: designed to fit the 256 KiB / 64 KiB chip from the start."""
    return keras.Sequential([
        keras.Input((n, 1)),
        layers.Conv1D(8, 7, strides=2, activation="relu"), layers.MaxPooling1D(2),
        layers.Conv1D(16, 7, activation="relu"), layers.MaxPooling1D(2),
        layers.Conv1D(32, 7, activation="relu"), layers.GlobalAveragePooling1D(),
        layers.Dense(1, activation="sigmoid")], name="small")


def large(n=625, keep=1.0):
    """1D ResNet: a stem conv, 4 residual blocks (2 convs each) and a dense head; 501,537 weights at keep=1.

    `keep` scales the width of every layer. keep<1 gives the narrower shape that structured pruning produces,
    so the same function builds both the pruned model and its train-from-scratch twin.
    Layers are named so that pruning can copy filters across by name.
    """
    w = lambda f: max(1, round(f * keep))
    x = inp = keras.Input((n, 1))
    x = layers.Conv1D(w(32), 15, strides=2, padding="same", activation="relu", name="c0")(x)
    for i, f in enumerate([32, 64, 96, 128], 1):
        y = layers.Conv1D(w(f), 9, padding="same", activation="relu", name=f"b{i}a")(x)
        y = layers.BatchNormalization(name=f"b{i}bn")(y)
        y = layers.Conv1D(w(f), 9, padding="same", name=f"b{i}b")(y)
        # shortcut: a 1x1 conv when the channel count changes, otherwise the input itself
        x = layers.Conv1D(w(f), 1, padding="same", name=f"b{i}s")(x) if x.shape[-1] != w(f) else x
        x = layers.MaxPooling1D(2)(layers.Activation("relu")(layers.Add()([x, y])))
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(layers.Dense(w(64), activation="relu", name="d1")(x))
    out = layers.Dense(1, activation="sigmoid", name="out")(x)
    return keras.Model(inp, out, name="large" if keep == 1 else f"large_k{int(keep * 100)}")


def count_macs(model):
    """Multiply-accumulate operations for one clip (conv and dense layers; pooling/adds are negligible)."""
    total = 0
    for l in model.layers:
        if isinstance(l, layers.Conv1D):
            k, cin, cout = l.kernel.shape
            total += l.output.shape[1] * k * cin * cout
        elif isinstance(l, layers.Dense):
            total += l.kernel.shape[0] * l.kernel.shape[1]
    return int(total)
