from keras import backend as K
from keras import activations, initializers, regularizers, constraints
from keras.layers import Layer
from keras.regularizers import l2
from keras.constraints import maxnorm

class QRNN(Layer):

    def __init__(self, units, window = 3, bias=True, stateful=False,
                kernel_init='uniform', kernel_reg=l2(1e-6), kernel_constr=maxnorm(5),
                bias_init='zero', bias_reg=l2(1e-6), bias_constr=maxnorm(5),
                input_dim=None, input_length=None, activation='tanh',
                ret_sequence = False, **kwargs):

        self.units = units
        self.window = window

        self.stateful = stateful
        self.ret_sequence = ret_sequence
        self.bias = bias

        self.kernel_init = kernel_init
        self.kernel_reg = kernel_reg
        self.kernel_constr = kernel_constr

        self.bias_init = bias_init
        self.bias_reg = bias_reg
        self.bias_constr = bias_constr

        self.activation = activations.get(activation)

        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        super(QRNN, self).__init__(**kwargs)


    def call(self, inputs):
        input_shape = K.int_shape(inputs)
        initial_states = [K.zeros((input_shape[0], self.units))]

        convoluted = K.conv1d(inputs, self.kernel, strides=1, padding='valid', data_format='channels_last')

        h, hs, states = K.rnn(self.step, convoluted, initial_states, constants = [], input_length=input_shape[1])

        return hs if self.ret_sequence else h

    def build(self, input_shape):
        self.input_dim = input_shape[2]

        self.kernel = self.add_weight(name='kernel',
                                    shape=(self.window, self.input_dim, self.units * 3),
                                    initializer=initializers.get(self.kernel_init),
                                    regularizer=regularizers.get(self.kernel_reg),
                                    constraint=constraints.get(self.kernel_constr)
                                    )
        if self.bias:
            self.bias = self.add_weight(name='bias', shape=(self.units*3,),
                                        initializer=initializers.get(self.bias_init),
                                        regularizer=regularizers.get(self.bias_reg),
                                        constraint=constraints.get(self.bias_constr))

        super(QRNN, self).build(input_shape)

    def step(self, inputs, states): #states should be (states x seq_length?) inputs should be (batch x (filters*input_dim))
        c_tminusone = states[0]

        Z = inputs[:, :self.units]
        F = inputs[:, self.units: 2*self.units]
        O = inputs[:, self.units*2:]

        z_t = self.activation(Z)
        f_t = K.sigmoid(F)
        o_t = K.sigmoid(O)

        c_t = f_t * c_tminusone + (1 - f_t) * z_t
        h_t = o_t * c_t

        return h_t, [h_t]

    def compute_output_shape(self, input_shape):
        if self.ret_sequence:
            return (input_shape[0], input_shape[1], self.units)
        else:
            return (input_shape[0], self.units)