from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Reshape, Dense, LSTM, GRU, Bidirectional, Lambda
import tensorflow.keras.backend as K
from tensorflow.keras.activations import softmax

def build_ocr_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Reshape to (timesteps, features) for RNN input
    rnn_input = Reshape(target_shape=((input_shape[0] // 4), (input_shape[1] // 4) * 64))(x)
    
    # Sequence modeling
    rnn_output = Bidirectional(LSTM(128, return_sequences=True))(rnn_input)
    rnn_output = Dense(num_classes, activation='softmax')(rnn_output)
    
    # CTC loss
    def ctc_loss_func(y_true, y_pred):
        return K.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    
    model = Model(inputs, rnn_output)
    model.compile(optimizer='adam', loss=ctc_loss_func)
    
    return model
