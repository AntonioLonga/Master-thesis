from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.initializers import RandomNormal

from spektral.layers import GraphAttention, GlobalAttentionPool
 

def gen_SpektralGNN_emb(n_classes,n_components,max_n_nodes,n_attributes):
    learning_rate = 0.001
    l2_reg = 5e-4  

    X_in = Input(shape=(max_n_nodes, n_attributes))
    filter_in = Input((max_n_nodes, max_n_nodes))
    emb_GNN = GraphAttention(32, activation='relu', kernel_regularizer=l2(l2_reg))([X_in, filter_in])
    emb_GNN = GraphAttention(n_components, activation='relu', kernel_regularizer=l2(l2_reg))([emb_GNN, filter_in])
    emb_GNN = GlobalAttentionPool(n_components)(emb_GNN)

    cla_GNN = Dense(n_classes, activation='softmax')(emb_GNN)

    optimizer = Adam(lr=learning_rate)
    classificator = Model(inputs=[X_in, filter_in], outputs=cla_GNN)
    embedder = Model(inputs=[X_in, filter_in], outputs=emb_GNN)
    classificator.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])
    
    return (classificator,embedder)


def gen_sup_auto_callback(w_dec,w_cla,input_auto,embedding_size=32):

    iniz = RandomNormal(mean=0, stddev=0.05)
    
    prob_drop = 0.5
    x_in = Input(shape=(input_auto,))
    enc = Dense(input_auto*2, activation="relu",kernel_initializer=iniz)(x_in)
    enc = Dropout(prob_drop)(enc)
    enc = Dense(input_auto, activation="relu",kernel_initializer=iniz)(enc)
    enc = Dense(embedding_size, activation="sigmoid",kernel_initializer=iniz)(enc)

    decode = Dense(input_auto, activation="relu",name='decoder',kernel_initializer=iniz)(enc)

    cla = Dense(10, activation='relu', name='cla_10')(enc)
    cla = Dense(2, activation='softmax',name='classifier')(cla)

    
    encoder = Model(x_in,enc)
    autoencoder = Model(inputs = x_in, outputs = [decode,cla])
    autoencoder.compile(optimizer='adam',
                                metrics={'decoder': 'mse', 'classifier': ['acc']},
                                loss = {'decoder': 'mean_squared_error', 'classifier': 'categorical_crossentropy'},
                                loss_weights = {'decoder': w_dec, 'classifier': w_cla})
    return(autoencoder,encoder)


def gen_sup_auto_callback_emb(w_dec,w_cla,input_auto,embedding_size=2):
    iniz = RandomNormal(mean=0, stddev=0.05)

    prob_drop = 0.5
    x_in = Input(shape=(input_auto,))
    enc = Dense(input_auto*2, activation="relu",name='enc_1',kernel_initializer=iniz)(x_in)
    enc = Dropout(prob_drop)(enc)
    enc = Dense(input_auto, activation="relu",name='enc_2',kernel_initializer=iniz)(enc)
    enc = Dense(50, activation="sigmoid",name='encoder',kernel_initializer=iniz)(enc)

    decode = Dense(int(input_auto*2), activation="relu",name='dec_1',kernel_initializer=iniz)(enc)
    decode = Dense(input_auto, activation="relu",name='decoder',kernel_initializer=iniz)(decode)

    emb = Dense(10,activation="relu",name='emb_1')(enc)
    emb = Dense(2,activation="relu",name="embedder")(emb)
    cla = Dense(5, activation='relu', name='cla_1')(emb)
    cla = Dense(2, activation='softmax',name='classifier')(cla)


    encoder = Model(x_in,enc)
    embedder = Model(x_in,emb)
    autoencoder = Model(inputs = x_in, outputs = [decode,cla])
    autoencoder.compile(optimizer='adam',
                                metrics={'decoder': 'mse', 'classifier': ['acc']},
                                loss = {'decoder': 'mean_squared_error', 'classifier': 'categorical_crossentropy'},
                                loss_weights = {'decoder': w_dec, 'classifier': w_cla})
    
    return(autoencoder,embedder)