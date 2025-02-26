from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

def get_callbacks(save_weights_path='./model/weights/best_model.keras', RLR_patience=2):
    ''' Callbacks for model training
    Arguments:
        save_weights_path: path to save best model weights
    '''
    checkpoint_callback = ModelCheckpoint(save_weights_path, monitor='val_acc', save_best_only=True, mode='max')
    rop_callback = ReduceLROnPlateau(monitor='val_loss', patience=RLR_patience)
    return [checkpoint_callback, rop_callback]
