import tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn_attention_deep(input_shape, num_classes):
    inp = layers.Input(shape=input_shape)
    x = layers.Conv1D(128, 3, padding="same")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv1D(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv1D(256, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv1D(256, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.2)(x)

    attn_scores = layers.Dense(1)(x)
    attn_scores = layers.Softmax(axis=1)(attn_scores)

    def weighted_sum(args):
        feats, scores = args
        return tf.reduce_sum(feats * scores, axis=1)

    x = layers.Lambda(weighted_sum)([x, attn_scores])
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    out = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inp, out)

def build_bilstm_attention(input_shape, num_classes):
    inp = layers.Input(shape=input_shape)

    x = layers.Conv1D(128, 3, padding="same")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(0.25)(x)

    mha = layers.MultiHeadAttention(num_heads=4, key_dim=64, dropout=0.2)
    attn_output = mha(query=x, value=x)
    x = layers.LayerNormalization()(x + attn_output)

    ffn = layers.Dense(256, activation="relu")(x)
    ffn = layers.Dropout(0.3)(ffn)
    ffn = layers.Dense(128)(ffn)
    x = layers.LayerNormalization()(x + ffn)

    avg_pool = layers.GlobalAveragePooling1D()(x)
    max_pool = layers.GlobalMaxPooling1D()(x)
    x = layers.Concatenate()([avg_pool, max_pool])

    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    out = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inp, out)

def build_transformer_model(input_shape, num_classes):
    inp = layers.Input(shape=input_shape)

    x = layers.Dense(128)(inp)
    x = layers.LayerNormalization()(x)

    positions = tf.range(start=0, limit=input_shape[0], delta=1)
    pos_emb = layers.Embedding(input_dim=input_shape[0], output_dim=128)(positions)
    x = x + pos_emb

    for _ in range(3):
        mha = layers.MultiHeadAttention(num_heads=4, key_dim=32, dropout=0.2)
        attn_output = mha(query=x, value=x, key=x)
        x = layers.LayerNormalization()(x + attn_output)

        ffn = tf.keras.Sequential([
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(128),
        ])
        ffn_output = ffn(x)
        x = layers.LayerNormalization()(x + ffn_output)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inp, out)

def build_resnet_1d(input_shape, num_classes):
    inp = layers.Input(shape=input_shape)

    def residual_block(x, filters, kernel_size=3):
        shortcut = x
        x = layers.Conv1D(filters, kernel_size, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Conv1D(filters, kernel_size, padding="same")(x)
        x = layers.BatchNormalization()(x)
        if shortcut.shape[-1] != filters:
            shortcut = layers.Conv1D(filters, 1, padding="same")(shortcut)
        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)
        return x

    x = layers.Conv1D(64, 7, padding="same")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = residual_block(x, 64)
    x = residual_block(x, 128)
    x = residual_block(x, 256)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inp, out)

def build_model_by_name(model_name: str, input_shape, num_classes):
    if model_name.startswith("Transformer"):
        return build_transformer_model(input_shape, num_classes)
    if model_name.startswith("ResNet1D"):
        return build_resnet_1d(input_shape, num_classes)
    if model_name.startswith("BiLSTM"):
        return build_bilstm_attention(input_shape, num_classes)
    if model_name.startswith("CNN_Attention"):
        return build_cnn_attention_deep(input_shape, num_classes)
    raise ValueError(f"알 수 없는 MODEL_NAME: {model_name}")
