from tensorflow.keras.layers import Input,Concatenate, Dense, Dropout, TimeDistributed, Reshape, Conv1D, GlobalMaxPooling1D, MaxPool2D
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import AUC
from tensorflow.keras.applications.vgg19 import VGG19

def get_vgg_model():
    """
    Transfer learning model for spatial feature extraction.
    returns: tf.keras.Model
    """
    transfer_model = VGG19(weights="imagenet", input_shape=(128,128,3),include_top = False)
    for layer in transfer_model.layers[:]:
        layer.trainable = False
    return transfer_model
    
def get_conv_model(kernel_size,dropout):
    """
    Temporal convolution model for temporal feature extraction.
    kernel_size: number of kernel for the 1D convolution, int
    dropout: to be applied after convolution and pooling, float (0,1)
    returns: tf.keras.Model
    """
    input_seq = Input(shape=(32, 512))
    n_filters = 3 #number of convolution filters
    convolved = Conv1D(n_filters, kernel_size, padding="same", activation="relu")(input_seq)
    processed = Dropout(dropout)(convolved)
    pooled = GlobalMaxPooling1D()(processed)
    compressed = Dropout(dropout)(pooled)
    conv_model = Model(inputs=input_seq, outputs=compressed)
    return conv_model

def run_main_model(input_2d_timeseries, dropout):
    """
    Main model, combines spatial and temporal feature extraction.
    returns: tf.keras.Model
    """
    # spatial feature extraction
    vgg_out = TimeDistributed(get_vgg_model())(input_2d_timeseries)
    vgg_out = TimeDistributed(MaxPool2D(pool_size=(4,4)))(vgg_out)
    vgg_out = Reshape(target_shape=(32,512))(vgg_out)
    vgg_out = Dropout(dropout)(vgg_out)
    # temporal feature extraction
    conv_net_small = get_conv_model(kernel_size=3, dropout=dropout, pooling=pooling)
    conv_net_large = get_conv_model(kernel_size=11,dropout=dropout, pooling=pooling)
    embedding_small = conv_net_small(vgg_out)
    embedding_large = conv_net_large(vgg_out)
    #concatenate all outputs
    merged = Concatenate()([embedding_small, embedding_large])
    return merged

if __name__ == "__main__":
    # define input
    inputs = Input(shape=(32,128,128,3,2))
    input_slice_low=inputs[:,:,:,:,:,0]
    input_slice_high=inputs[:,:,:,:,:,1]
    # run model
    output_slice_low = run_main_model(input_slice_low, dropout)
    output_slice_high = run_main_model(input_slice_high, dropout)
    #merge outputs for both slices
    merged=Concatenate()([output_slice_low,output_slice_high])
    # fc and classify
    fcn = Dense(32, activation="relu")(merged)
    out = Dense(1, activation='sigmoid')(fcn)
    # compile model
    model = Model(inputs=inputs, outputs=out)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', 
                  metrics=[AUC(num_thresholds=1000)])