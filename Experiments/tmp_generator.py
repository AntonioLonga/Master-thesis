from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras import losses
from keras.optimizers import RMSprop
from keras.initializers import RandomNormal


from embedder import Transformer
from embedder import Transformer_autoencoder
from embedder import Transformer_GNN
from embedder import Transformer_GNN_embedder
from embedder import Preprocessing_scaler

from spektral.layers import GraphAttention, GlobalAttentionPool
from spektral.utils import localpooling_filter




#################### Autoencoder 128 - 64- 16 - 8 - n_components
def gen_autoencoder(n_components):
     
    #autoencoder
    iniz = RandomNormal(mean=0.0, stddev=0.1)

    X_in = Input(shape=(128,))
    dense_64 = Dense(64, activation='relu',kernel_initializer=iniz)(X_in)
    dense_16 = Dense(16, activation='relu',kernel_initializer=iniz)(dense_64)
    dense_8 = Dense(8, activation='relu',kernel_initializer=iniz)(dense_16)
    dense_x = Dense(n_components, activation='softmax',kernel_initializer=iniz)(dense_8)
    dense_8_2 = Dense(8, activation='relu',kernel_initializer=iniz)(dense_x)
    dense_16_2 = Dense(16, activation='relu',kernel_initializer=iniz)(dense_8_2)
    dense_64_2 = Dense(64, activation='relu',kernel_initializer=iniz)(dense_16_2)
    out = Dense(128,  activation='relu',kernel_initializer=iniz)(dense_64_2)
    model = Model(inputs=X_in, outputs=out)
    model.compile(optimizer=RMSprop(), loss=losses.mean_squared_error)

    
    
    #encoder
    X_in_enc = Input(shape=(128,))
    dense_64_enc = Dense(64, activation='relu')(X_in_enc)
    dense_16_enc = Dense(16, activation='relu')(dense_64_enc)
    dense_8_enc = Dense(8, activation='relu')(dense_16_enc)
    dense_x_enc = Dense(n_components, activation='relu')(dense_8_enc)
    model_enc = Model(inputs=[X_in_enc], outputs=dense_x_enc)
    model_enc.compile(optimizer=RMSprop(), loss=losses.mean_squared_error)
    
    return (model, model_enc)

def gen_transf_autoencoder(batch_size,validation_split,epochs,verbose=0,scaler=None,dim=[2,3,5],plot=False):
    
    autoencoder = []
    encoder = []
    for d in dim:
        
        a, e = gen_autoencoder(d)
        
        autoencoder.append(a)
        encoder.append(e)
        
   
    transf = Transformer_autoencoder(autoencoder,encoder,batch_size,validation_split,epochs,verbose,scaler,dim,plot)
    
    return(transf)
    

    


################# genera GNN  (fino a 128)   
def generate_GNN(max_n_nodes,n_attributes,n_classes,batch_size = 32,
                 validation_split = 0.1,epochs = 100,verbose=0, plot=False):
    

    learning_rate = 0.001
    l2_reg = 5e-4  

    ##### DEFINISCI MODELLO ORIGINALE
    X_in_1_1 = Input(shape=(max_n_nodes, n_attributes))
    filter_in_1_1 = Input((max_n_nodes, max_n_nodes))
    gc1_1_1 = GraphAttention(32, activation='relu', kernel_regularizer=l2(l2_reg))([X_in_1_1, filter_in_1_1])
    gc2_1_1 = GraphAttention(32, activation='relu', kernel_regularizer=l2(l2_reg))([gc1_1_1, filter_in_1_1])
    pool_1_1 = GlobalAttentionPool(128)(gc2_1_1)
    output_1_1 = Dense(n_classes, activation='softmax')(pool_1_1)
    model_1_1 = Model(inputs=[X_in_1_1, filter_in_1_1], outputs=output_1_1)
    optimizer = Adam(lr=learning_rate)
    model_1_1.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])


    ##### CREA IL SECONDO MODELLO
    X_in_1_2 = Input(shape=(max_n_nodes, n_attributes))
    filter_in_1_2 = Input((max_n_nodes, max_n_nodes))
    gc1_1_2 = GraphAttention(32, activation='relu', kernel_regularizer=l2(l2_reg))([X_in_1_2, filter_in_1_2])
    gc2_1_2 = GraphAttention(32, activation='relu', kernel_regularizer=l2(l2_reg))([gc1_1_2, filter_in_1_2])
    pool_1_2 = GlobalAttentionPool(128)(gc2_1_2)
    model_1_2 = Model(inputs=[X_in_1_2, filter_in_1_2], outputs=pool_1_2)
    model_1_2.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=['acc'])


    my_GNN_1 = Transformer_GNN(original_model = model_1_1,
                                      new_model = model_1_2,
                                      batch_size = batch_size,
                                      validation_split = validation_split,
                                      epochs = epochs,
                                      verbose=verbose,
                                      plot=plot)
    
    
    return(my_GNN_1)
    





#################### genera gnn + Dense layer 128 - 64- 32 -16 -8 - 5 - 3 - 2
def add_layer(input_layer,out):
    
    layer = Dense(out)(input_layer)
    return(layer)
    
def gen_dense(n_components,max_n_nodes, n_attributes):
    
    layers = [64,32,16,8,5,3,2]
    
    learning_rate = 0.001
    l2_reg = 5e-4
    n_classes = 2

    # origlinale
    X_in_1_1 = Input(shape=(max_n_nodes, n_attributes))
    filter_in_1_1 = Input((max_n_nodes, max_n_nodes))
    gc1_1_1 = GraphAttention(32, activation='relu', kernel_regularizer=l2(l2_reg))([X_in_1_1, filter_in_1_1])
    gc2_1_1 = GraphAttention(32, activation='relu', kernel_regularizer=l2(l2_reg))([gc1_1_1, filter_in_1_1])
    pool_1_1 = GlobalAttentionPool(128)(gc2_1_1)

    index = (layers.index(n_components)) + 1
    input_layer = pool_1_1

    for i in range(0,index):
        layer = add_layer(input_layer, layers[i])
        input_layer = layer

    output_1_1 = Dense(n_classes, activation='softmax')(input_layer)
    model_1_1 = Model(inputs=[X_in_1_1, filter_in_1_1], outputs=output_1_1)
    optimizer = Adam(lr=learning_rate)
    model_1_1.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])
    
    
    # embedder
    
    X_in_2= Input(shape=(max_n_nodes, n_attributes))
    filter_in_2 = Input((max_n_nodes, max_n_nodes))
    gc1_2 = GraphAttention(32, activation='relu', kernel_regularizer=l2(l2_reg))([X_in_2, filter_in_2])
    gc2_2 = GraphAttention(32, activation='relu', kernel_regularizer=l2(l2_reg))([gc1_2, filter_in_2])
    pool_2 = GlobalAttentionPool(128)(gc2_2)

    
    index = (layers.index(n_components)) + 1
    input_layer = pool_2
    
    for i in range(0,index):
        layer = add_layer(input_layer, layers[i])
        input_layer = layer

        
    model_2 = Model(inputs=[X_in_2, filter_in_2], outputs=input_layer)
    optimizer = Adam(lr=learning_rate)
    model_2.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])
    
    
    return(model_1_1,model_2)

def  gen_gnn_dense(max_n_nodes, n_attributes,batch_size,validation_split,epochs,dim,plot=False):

    n_classes = 2
    learning_rate = 0.001
    l2_reg = 5e-4  
    
    
    original = []
    encoder = []
    
    for d in dim:
        ori, enc = gen_dense(d,max_n_nodes, n_attributes)
        original.append(ori)
        encoder.append(enc)
    


    my_GNN_1 = Transformer_GNN_embedder(original_models = original,
                                      new_models = encoder,
                                      batch_size = batch_size,
                                      validation_split = validation_split,
                                      epochs = epochs,
                                      plot = plot)

    return(my_GNN_1)





#################### genera gnn + Dense layer 128 - n_components

def gen_small(max_n_nodes, n_attributes,n_components):
    n_classes = 2

    learning_rate = 0.001
    l2_reg = 5e-4  
    
    ##### DEFINISCI MODELLO ORIGINALE
    X_in_1_1 = Input(shape=(max_n_nodes, n_attributes))
    filter_in_1_1 = Input((max_n_nodes, max_n_nodes))
    gc1_1_1 = GraphAttention(32, activation='relu', kernel_regularizer=l2(l2_reg))([X_in_1_1, filter_in_1_1])
    gc2_1_1 = GraphAttention(32, activation='relu', kernel_regularizer=l2(l2_reg))([gc1_1_1, filter_in_1_1])
    pool_1_1 = GlobalAttentionPool(128)(gc2_1_1)
    dense_x = Dense(n_components)(pool_1_1)
    output_1_1 = Dense(n_classes, activation='softmax')(dense_x)
    model_1_1 = Model(inputs=[X_in_1_1, filter_in_1_1], outputs=output_1_1)
    optimizer = Adam(lr=learning_rate)
    model_1_1.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])


    ##### CREA IL SECONDO MODELLO
    X_in_1_2 = Input(shape=(max_n_nodes, n_attributes))
    filter_in_1_2 = Input((max_n_nodes, max_n_nodes))
    gc1_1_2 = GraphAttention(32, activation='relu', kernel_regularizer=l2(l2_reg))([X_in_1_2, filter_in_1_2])
    gc2_1_2 = GraphAttention(32, activation='relu', kernel_regularizer=l2(l2_reg))([gc1_1_2, filter_in_1_2])
    pool_1_2 = GlobalAttentionPool(128)(gc2_1_2)
    dense2_x = Dense(n_components)(pool_1_2)
    model_1_2 = Model(inputs=[X_in_1_2, filter_in_1_2], outputs=dense2_x)
    model_1_2.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=['acc'])
    
    
    return(model_1_1,model_1_2)

def  gen_gnn_small(max_n_nodes, n_attributes,batch_size,validation_split,epochs,dim,verbose=False,plot=False):


    original_models = []
    new_models = []
    
    for d in dim:
        original, new = gen_small(max_n_nodes, n_attributes, d)
        
        original_models.append(original)
        new_models.append(new)
    
    
    my_GNN_1 = Transformer_GNN_embedder(original_models = original_models,
                                      new_models = new_models,
                                      batch_size = batch_size,
                                      validation_split = validation_split,
                                      epochs = epochs,
                                      verbose = verbose,
                                      plot = plot)

    return(my_GNN_1)