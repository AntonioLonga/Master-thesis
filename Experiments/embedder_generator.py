from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.initializers import RandomNormal

from spektral.layers import GraphAttention, GlobalAttentionPool
 

from eden import graph
import embedder
import umap
from sklearn.decomposition import TruncatedSVD

from keras import backend as K
from sklearn.preprocessing import Normalizer
from my_callbacks import MyCallback_sinusoidal



from keras.callbacks import TensorBoard
from datetime import datetime


def gen_vec_supAutoEmb(n_classes,auto_n_components,emb_n_components,
                        sauto_batch_size=32,
                        sauto_validation_split=0.2,
                        sauto_epochs=500,
                        sauto_plateau=10,
                        sauto_k=1,
                        sauto_n_period=4,
                        sauto_scale_c=0.1,
                        sauto_scale_d=0.1,
                        tb_folder = None):
    #### kernel
    k1 = embedder.Transformer(graph.Vectorizer(complexity=5,nbits=16), has_fit = False)

    ###### !!!!!! ATTENZIONE !!!!!!
    #### n_components < len(train_set)
    m1 = embedder.Transformer(TruncatedSVD(n_components=450))


    #### from kernel to medium
    w_dec = K.variable(1)
    w_cla = K.variable(1)
    auto,enc = gen_sup_auto_callback_emb(w_dec,w_cla,450,auto_n_components,emb_n_components,n_classes)

    # CALLBACK  
    my_call_sin = MyCallback_sinusoidal(w_dec, w_cla, sauto_epochs,sauto_plateau,sauto_k,sauto_n_period,sauto_scale_c,sauto_scale_d)

    # SCALER
    scaler = embedder.Preprocessing_scaler([0, 1])    
    # NORMALIZER
    normalizer = Normalizer(copy=True, norm='l2')


 
    if (tb_folder == None):
        tb2 = [my_call_sin]
    else:
        now = datetime.now()
        current_time = now.strftime("%H_%M_%S")
        path = tb_folder + "/supAutoEmb_"+current_time
        tb2 = [my_call_sin,TensorBoard(log_dir=path)]

    m2 = embedder.Transformer_sup_autoencoder(auto,enc,
                                             batch_size=sauto_batch_size,
                                             validation_split=sauto_validation_split,
                                             epochs=sauto_epochs,
                                             callbacks=tb2,
                                             verbose=0,
                                             normal=normalizer,
                                             scaler=scaler)
    m2 = embedder.Transformer(m2)

    
    name = "Vect PCA-450 SupAuto("
    name = name+str(sauto_epochs)+")-"+str(auto_n_components)+" EMB-"+str(emb_n_components)
    emb = embedder.Embedder([k1,m1,m2],name=name)

    return(emb)




def gen_spk_supAutoEmb(n_classes,max_n_nodes,n_attributes,spk_n_components,auto_n_components,emb_n_components,
                        spk_batch_size = 32,
                        spk_validation_split = 0.2,
                        spk_epochs = 10,
                        spk_patience = 20,
                        sauto_batch_size=32,
                        sauto_validation_split=0.2,
                        sauto_epochs=500,
                        sauto_plateau=10,
                        sauto_k=1,
                        sauto_n_period=4,
                        sauto_scale_c=0.1,
                        sauto_scale_d=0.1,
                        tb_folder = None):
    #### kernel
    spk_gnn, spk_emb = gen_SpektralGNN_emb(n_classes= n_classes,
                                            n_components= spk_n_components,
                                            max_n_nodes= max_n_nodes,
                                            n_attributes= n_attributes)

    if (tb_folder == None):
        tb1 = None
    else:
        now = datetime.now()
        current_time = now.strftime("%H_%M_%S")
        path = tb_folder + "/spek_"+current_time
        tb1 = [TensorBoard(log_dir=path)]
    
    kernel_spk = embedder.Kernel_GNN(classificator = spk_gnn,
                                    embedder = spk_emb,
                                    batch_size = spk_batch_size,
                                    validation_split = spk_validation_split,
                                    callbacks= tb1,
                                    epochs = spk_epochs,
                                    patience = spk_patience)

    k1 = embedder.Transformer(kernel_spk)

    #### from kernel to medium
    w_dec = K.variable(1)
    w_cla = K.variable(1)
    auto,enc = gen_sup_auto_callback_emb(w_dec,w_cla,spk_n_components,auto_n_components,emb_n_components,n_classes)

    # CALLBACK  
    my_call_sin = MyCallback_sinusoidal(w_dec, w_cla, sauto_epochs,sauto_plateau,sauto_k,sauto_n_period,sauto_scale_c,sauto_scale_d)

    # SCALER
    scaler = embedder.Preprocessing_scaler([0, 1])    
    # NORMALIZER
    normalizer = Normalizer(copy=True, norm='l2')


 
    if (tb_folder == None):
        tb2 = [my_call_sin]
    else:
        now = datetime.now()
        current_time = now.strftime("%H_%M_%S")
        path = tb_folder + "/supAutoEmb_"+current_time
        tb2 = [my_call_sin,TensorBoard(log_dir=path)]

    m1 = embedder.Transformer_sup_autoencoder(auto,enc,
                                             batch_size=sauto_batch_size,
                                             validation_split=sauto_validation_split,
                                             epochs=sauto_epochs,
                                             callbacks=tb2,
                                             verbose=0,
                                             normal=normalizer,
                                             scaler=scaler)
    m1 = embedder.Transformer(m1)

    
    name = "Spk("+str(spk_epochs)+")-"+str(spk_n_components)+" SupAuto("
    name = name+str(sauto_epochs)+")-"+str(auto_n_components)+" EMB-"+str(emb_n_components)
    emb = embedder.Embedder([k1,m1],name=name)

    return(emb)




def gen_vect_supAuto_dnn(auto_n_components,emb_n_components,
                        sauto_batch_size=32,
                        sauto_validation_split=0.2,
                        sauto_epochs=50,
                        sauto_plateau=10,
                        sauto_k=1,
                        sauto_n_period=2,
                        sauto_scale_c=0.1,
                        sauto_scale_d=0.1,
                        dnn_epochs=50,
                        dnn_batch_size=32):
    #### kernel
    k1 = embedder.Transformer(graph.Vectorizer(complexity=5,nbits=16), has_fit = False)

    ###### !!!!!! ATTENZIONE !!!!!!
    #### n_components < len(train_set)
    m1 = embedder.Transformer(TruncatedSVD(n_components=450))

    #### from kernel to medium
    w_dec = K.variable(1)
    w_cla = K.variable(1)
    auto,enc = gen_sup_auto_callback(w_dec,w_cla,450,embedding_size=auto_n_components)

    # CALLBACK  
    my_call_sin = MyCallback_sinusoidal(w_dec, w_cla, sauto_epochs,sauto_plateau,sauto_k,sauto_n_period,sauto_scale_c,sauto_scale_d)

    # SCALER
    scaler = embedder.Preprocessing_scaler([0, 1])    
    # NORMALIZER
    normalizer = Normalizer(copy=True, norm='l2')


    now = datetime.now()
    current_time = now.strftime("%H_%M_%S")
    path = "vec_sauto_dnn/auto_"+current_time
    tb2 = TensorBoard(log_dir=path)
     

    m2 = embedder.Transformer_sup_autoencoder(auto,enc,
                                             batch_size=sauto_batch_size,
                                             validation_split=sauto_validation_split,
                                             epochs=sauto_epochs,
                                             callbacks=[my_call_sin,tb2],
                                             verbose=0,
                                             normal=normalizer,
                                             scaler=scaler)
    m2 = embedder.Transformer(m2)

    ##### from medium to low
    x_in = Input(shape=(auto_n_components,))
    den = Dense(int(auto_n_components/2),activation="relu")(x_in)
    den = Dense(int(auto_n_components/4),activation="relu")(den)
    den = Dense(emb_n_components,activation="sigmoid")(den)

    dnn = Model(x_in,den)
    dnn.compile(optimizer="adam",loss="MSE")

    uma = umap.UMAP(n_components=emb_n_components)

    now = datetime.now()
    current_time = now.strftime("%H_%M_%S")
    path = "vec_sauto_dnn/dnn_"+current_time
    tb3 = TensorBoard(log_dir=path)
    

    dnn_umap = embedder.Transformer_DNN_umap(dnn,uma,
                                            epochs=dnn_epochs,
                                            batch_size=dnn_batch_size,
                                            callbacks = [tb3])

    m3 = embedder.Transformer(dnn_umap)

    name = "Vect PCA-450 SupAuto("
    name = name+str(sauto_epochs)+")-"+str(auto_n_components)+" UMAP-"+str(emb_n_components)
    emb = embedder.Embedder([k1,m1,m2,m3],name=name)

    return(emb)




def gen_spk_supAuto_dnn(n_classes,max_n_nodes,n_attributes,spk_n_components,auto_n_components,emb_n_components,
                        spk_batch_size = 32,
                        spk_validation_split = 0.2,
                        spk_epochs = 10,
                        spk_patience = 20,
                        sauto_batch_size=32,
                        sauto_validation_split=0.2,
                        sauto_epochs=50,
                        sauto_plateau=10,
                        sauto_k=1,
                        sauto_n_period=2,
                        sauto_scale_c=0.1,
                        sauto_scale_d=0.1,
                        dnn_epochs=50,
                        dnn_batch_size=32):
    #### kernel
    spk_gnn, spk_emb = gen_SpektralGNN_emb(n_classes= n_classes,
                                            n_components= spk_n_components,
                                            max_n_nodes= max_n_nodes,
                                            n_attributes= n_attributes)


    now = datetime.now()
    current_time = now.strftime("%H_%M_%S")
    path = "spk_sauto_dnn/kernel_"+current_time
    tb1 = TensorBoard(log_dir=path)
    
    kernel_spk = embedder.Kernel_GNN(classificator = spk_gnn,
                                    embedder = spk_emb,
                                    batch_size = spk_batch_size,
                                    validation_split = spk_validation_split,
                                    callbacks = [tb1],
                                    epochs = spk_epochs,
                                    patience = spk_patience)

    k1 = embedder.Transformer(kernel_spk)

    #### from kernel to medium
    w_dec = K.variable(1)
    w_cla = K.variable(1)
    auto,enc = gen_sup_auto_callback(w_dec,w_cla,spk_n_components,embedding_size=auto_n_components)

    # CALLBACK  
    my_call_sin = MyCallback_sinusoidal(w_dec, w_cla, sauto_epochs,sauto_plateau,sauto_k,sauto_n_period,sauto_scale_c,sauto_scale_d)

    # SCALER
    scaler = embedder.Preprocessing_scaler([0, 1])    
    # NORMALIZER
    normalizer = Normalizer(copy=True, norm='l2')


    now = datetime.now()
    current_time = now.strftime("%H_%M_%S")
    path = "spk_sauto_dnn/auto_"+current_time
    tb2 = TensorBoard(log_dir=path)
     

    m1 = embedder.Transformer_sup_autoencoder(auto,enc,
                                             batch_size=sauto_batch_size,
                                             validation_split=sauto_validation_split,
                                             epochs=sauto_epochs,
                                             callbacks=[my_call_sin,tb2],
                                             verbose=0,
                                             normal=normalizer,
                                             scaler=scaler)
    m1 = embedder.Transformer(m1)

    ##### from medium to low
    x_in = Input(shape=(auto_n_components,))
    den = Dense(int(auto_n_components/2),activation="relu")(x_in)
    den = Dense(int(auto_n_components/4),activation="relu")(den)
    den = Dense(emb_n_components,activation="sigmoid")(den)

    dnn = Model(x_in,den)
    dnn.compile(optimizer="adam",loss="MSE")

    uma = umap.UMAP(n_components=emb_n_components)

    now = datetime.now()
    current_time = now.strftime("%H_%M_%S")
    path = "spk_sauto_dnn/dnn_"+current_time
    tb3 = TensorBoard(log_dir=path)
    

    dnn_umap = embedder.Transformer_DNN_umap(dnn,uma,
                                            epochs=dnn_epochs,
                                            batch_size=dnn_batch_size,
                                            callbacks = [tb3])

    m2 = embedder.Transformer(dnn_umap)

    name = "Spk("+str(spk_epochs)+")-"+str(spk_n_components)+" SupAuto("
    name = name+str(sauto_epochs)+")-"+str(auto_n_components)+" UMAP-"+str(emb_n_components)
    emb = embedder.Embedder([k1,m1,m2],name=name)

    return(emb)



def gen_baseline_spk(n_classes,max_n_nodes,n_attributes,spk_n_components=128,batch_size=32,
                    validation_split=0.2,epochs=300,patience=50,emb_n_components=2):
    spk_gnn, spk_emb = gen_SpektralGNN_emb(n_classes= n_classes,
                                            n_components= spk_n_components,
                                            max_n_nodes= max_n_nodes,
                                            n_attributes= n_attributes)

    kernel_spk = embedder.Kernel_GNN(classificator = spk_gnn,
                                    embedder = spk_emb,
                                    batch_size = batch_size,
                                    validation_split = validation_split,
                                    epochs = epochs,
                                    patience = patience)

    k1 = embedder.Transformer(kernel_spk)
    # umap
    m1 = embedder.Transformer(umap.UMAP(n_components = emb_n_components))
    # embedder
    name = "BASELINE: Spk("+str(epochs)+")-"+str(spk_n_components)+" UMAP-"+str(emb_n_components)
    emb = embedder.Embedder([k1,m1],name)
    return(emb)


def gen_baseline_vectorize(complexity=5,pca_n_components = 1000, emb_n_components=2):
    vetcoriz = graph.Vectorizer(complexity = complexity)
    m_1= embedder.Transformer(vetcoriz, has_fit = False)
    pca = TruncatedSVD(n_components=pca_n_components)
    m_2 = embedder.Transformer(pca)
    uma = umap.UMAP(n_components = emb_n_components)
    m_3 = embedder.Transformer(uma)
    name = "BASELINE: Vec-"+str(complexity)+" PCA-"+str(pca_n_components)+" UMAP-"+str(emb_n_components)
    emb = embedder.Embedder([m_1,m_2,m_3], name=name)
    
    return(emb)



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


def gen_sup_auto_callback(w_dec,w_cla,input_auto,embedding_size=32,n_classes=2):

    iniz = RandomNormal(mean=0, stddev=0.05)
    
    prob_drop = 0.5
    x_in = Input(shape=(input_auto,))
    enc = Dense(input_auto*2, activation="relu",kernel_initializer=iniz)(x_in)
    enc = Dropout(prob_drop)(enc)
    enc = Dense(input_auto, activation="relu",kernel_initializer=iniz)(enc)
    enc = Dense(embedding_size, activation="sigmoid",kernel_initializer=iniz)(enc)

    decode = Dense(input_auto, activation="relu",name='decoder',kernel_initializer=iniz)(enc)

    cla = Dense(10, activation='relu', name='cla_10')(enc)
    cla = Dense(n_classes, activation='softmax',name='classifier')(cla)

    
    encoder = Model(x_in,enc)
    autoencoder = Model(inputs = x_in, outputs = [decode,cla])
    autoencoder.compile(optimizer='adam',
                                metrics={'decoder': 'mse', 'classifier': ['acc']},
                                loss = {'decoder': 'mean_squared_error', 'classifier': 'categorical_crossentropy'},
                                loss_weights = {'decoder': w_dec, 'classifier': w_cla})
    return(autoencoder,encoder)


def gen_sup_auto_callback_emb(w_dec,w_cla,input_auto,auto_n_components,emb_n_components,n_classes):
    iniz = RandomNormal(mean=0, stddev=0.05)

    prob_drop = 0.5
    x_in = Input(shape=(input_auto,))
    enc = Dense(input_auto*2, activation="relu",name='enc_1',kernel_initializer=iniz)(x_in)
    enc = Dropout(prob_drop)(enc)
    enc = Dense(input_auto, activation="relu",name='enc_2',kernel_initializer=iniz)(enc)
    enc = Dense(auto_n_components, activation="sigmoid",name='encoder',kernel_initializer=iniz)(enc)

    decode = Dense(int(input_auto/2), activation="relu",name='dec_1',kernel_initializer=iniz)(enc)
    decode = Dense(input_auto, activation="relu",name='decoder',kernel_initializer=iniz)(decode)

    emb = Dense(10,activation="relu",name='emb_1')(enc)
    emb = Dense(emb_n_components,activation="relu",name="embedder")(emb)
    cla = Dense(20  , activation='relu', name='cla_1')(emb)
    cla = Dense(n_classes, activation='softmax',name='classifier')(cla)


    encoder = Model(x_in,enc)
    embedder = Model(x_in,emb)
    autoencoder = Model(inputs = x_in, outputs = [decode,cla])
    autoencoder.compile(optimizer='adam',
                                metrics={'decoder': 'mse', 'classifier': ['acc']},
                                loss = {'decoder': 'mean_squared_error', 'classifier': 'categorical_crossentropy'},
                                loss_weights = {'decoder': w_dec, 'classifier': w_cla})
    
    return(autoencoder,embedder)