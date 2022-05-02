from io import BytesIO
from pathlib import Path as p

import carryless_rangecoder as rc
import numpy as np


def array2bytes(img: np.ndarray, info: np.ndarray, file=None):
    # 無情報ヒストグラム
    # オンライン学習？ジグザグサーチによる時系列解析？
    count = np.ones(256, dtype=int)
    count_cum = count.cumsum()

    out = BytesIO()
    with rc.Encoder(out) as enc:
        for pix in img.flatten().astype(np.uint8).tolist():
            enc.encode(count.tolist(), count_cum.tolist(), pix)
            count[pix] += 1
            count_cum[pix:] += 1

    encoded = out.getvalue() + bytes(info.flatten().tolist())
    if file:
        p(file).write_bytes(encoded)
    else:
        print("bytes:", len(encoded))
        print("dpp:", len(encoded) / 784 * 8)
    return encoded


def bytes2array(bytes_, file=None):
    img = bytes_[:-10]
    info = bytes_[-10:]

    count = np.ones(256, dtype=int)
    count_cum = count.cumsum()

    decoded = []
    with rc.Decoder(BytesIO(img)) as dec:
        for _ in range(28**2):
            pix = dec.decode(count.tolist(), count_cum.tolist())
            decoded.append(pix)
            count[pix] += 1
            count_cum[pix:] += 1

    return np.array(decoded).reshape(28, 28), np.array(list(info))


if __name__ == "__main__":
    img = np.random.randint(0, 255, size=[28, 28])
    info = np.random.randint(0, 255, size=10)
    out = array2bytes(img, info)
    img2, info2 = bytes2array(out)
    assert (img == img2).all()
    assert (info == info2).all()
