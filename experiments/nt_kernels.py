from neural_tangents import stax


__all__ = [
    "get_mlp_kernel",
    "get_cnn_kernel",
    "get_conv_resnet_kernel",
    "get_dense_resnet_kernel",
]


def get_act_class(act):
    if act == "relu":
        return stax.Relu
    elif act == "erf":
        return stax.Erf
    else:
        raise KeyError("Unsupported act '{}'".format(act))


def get_mlp_kernel(num_hiddens, num_class=1, act="relu", w_std=1., b_std=0., last_w_std=1.):
    act_class = get_act_class(act)

    layers = []
    for _ in range(num_hiddens):
        layers.append(stax.Dense(512, W_std=w_std, b_std=b_std))
        layers.append(act_class())
    layers.append(stax.Dense(num_class, W_std=last_w_std))

    _, _, kernel_fn = stax.serial(*layers)
    return  kernel_fn


def get_cnn_kernel(num_hiddens, num_class=1, act="relu", w_std=1., b_std=0., last_w_std=1.):
    act_class = get_act_class(act)

    layers = []
    for _ in range(num_hiddens):
        layers.append(stax.Conv(1, (3, 3), (1, 1), "SAME", W_std=w_std, b_std=b_std))
        layers.append(act_class())
    layers.append(stax.Flatten())
    layers.append(stax.Dense(num_class, W_std=last_w_std))

    _, _, kernel_fn = stax.serial(*layers)
    return  kernel_fn


def get_conv_resnet_kernel(num_hiddens, num_class, act="relu", w_std=1., b_std=0., last_w_std=1.):
    act_class = get_act_class(act)

    def WideResnetBlock(channels, strides=(1, 1), channel_mismatch=False):
        Main = stax.serial(
            act_class(), stax.Conv(channels, (3, 3), strides, padding="SAME", W_std=w_std, b_std=b_std),
            act_class(), stax.Conv(channels, (3, 3), padding="SAME", W_std=w_std, b_std=b_std))
        Shortcut = stax.Identity() if not channel_mismatch else stax.Conv(
            channels, (3, 3), strides, padding="SAME", W_std=w_std, b_std=b_std)
        return stax.serial(stax.FanOut(2),
                           stax.parallel(Main, Shortcut),
                           stax.FanInSum())

    def WideResnetGroup(n, channels, strides=(1, 1)):
        blocks = []
        blocks += [WideResnetBlock(channels, strides, channel_mismatch=True)]
        for _ in range(n - 1):
            blocks += [WideResnetBlock(channels, (1, 1))]
        return stax.serial(*blocks)

    def WideResnet(block_size, k, num_class):
        return stax.serial(
            stax.Conv(16, (3, 3), padding="SAME", W_std=w_std, b_std=b_std),
            WideResnetGroup(block_size, int(8 * k)),
            WideResnetGroup(block_size, int(16 * k), (2, 2)),
            WideResnetGroup(block_size, int(32 * k), (2, 2)),
            WideResnetGroup(block_size, int(64 * k), (2, 2)),
            # stax.AvgPool((8, 8)),
            stax.Flatten(),
            stax.Dense(num_class, W_std=last_w_std))

    _, _, kernel_fn = WideResnet(block_size=num_hiddens, k=1, num_class=num_class)
    return kernel_fn


def get_dense_resnet_kernel(num_hiddens, num_class=1, act="relu", w_std=1., b_std=0., last_w_std=1.):
    act_class = get_act_class(act)

    ResBlock = stax.serial(
        stax.FanOut(2),
        stax.parallel(
            stax.serial(
                act_class(),
                stax.Dense(512, W_std=w_std, b_std=b_std),
            ),
            stax.Identity()
        ),
        stax.FanInSum()
    )

    layers = [stax.Dense(512, W_std=w_std, b_std=b_std)]
    layers += [ResBlock for _ in range(num_hiddens)]
    layers += [act_class(), stax.Dense(num_class, W_std=last_w_std)]

    _, _, kernel_fn = stax.serial(*layers)
    return kernel_fn
