from keras.layers import Input, Lambda, Dense
from keras.models import Model
from keras.utils import plot_model
import numpy as np

def build():
    input = Input((1,), name="input")
    a = Lambda(lambda x: 2 * x[:, :], name='a')
    b = Lambda(lambda x: 2 * x[:, :], name='b')
    output = Lambda(lambda x: 2 * x[:, :], name='output')

    o = a(input)
    o = b(o)
    o = a(o)
    o = output(o)

    model = Model(input, o)
    model.compile()
    model.summary()
    return model


def main():
    model = build()
    plot_model(model, "my_first_model.png")
    data = np.reshape(np.ones(1), (1, 1))
    print(data)
    res = model.predict(data)
    print("Result is : {}".format(res))
    res = model.predict(data)
    print("Result is : {}".format(res))


if __name__ == '__main__':
    main()
