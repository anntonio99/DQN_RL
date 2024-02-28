import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from tensorflow import keras
from keras import regularizers

class MessagePassingNN(tf.keras.Model):
    def __init__(self, hparams):
        super(MessagePassingNN, self).__init__()
        self.hparams = hparams

        # Define layers here
        self.Message = tf.keras.models.Sequential()
        self.Message.add(keras.layers.Dense(self.hparams['hidden_dim'],
                                            activation=tf.nn.selu, name="FirstLayer"))

        self.Update = tf.keras.layers.GRUCell(self.hparams['hidden_dim'], dtype=tf.float32)

        self.Readout = tf.keras.models.Sequential()
        self.Readout.add(keras.layers.Dense(self.hparams['readout_units'],
                                            activation=tf.nn.selu,
                                            kernel_regularizer=regularizers.l2(hparams['l2']),
                                            name="Readout1"))
        self.Readout.add(keras.layers.Dropout(rate=hparams['dropout_rate']))
        self.Readout.add(keras.layers.Dense(self.hparams['readout_units'],
                                            activation=tf.nn.selu,
                                            kernel_regularizer=regularizers.l2(hparams['l2']),
                                            name="Readout2"))
        self.Readout.add(keras.layers.Dropout(rate=hparams['dropout_rate']))
        self.Readout.add(keras.layers.Dense(1, kernel_regularizer=regularizers.l2(hparams['l2']),
                                            name="Readout3"))

    def build(self, input_shape=None):
        self.Message.build(input_shape=tf.TensorShape([None, self.hparams['hidden_dim']*2]))
        self.Update.build(input_shape=tf.TensorShape([None,self.hparams['hidden_dim']]))
        self.Readout.build(input_shape=[None, self.hparams['hidden_dim']])
        self.built = True

    @tf.function
    def call(self, features, graph_ids, edges_topology, training=False):
        # Define the forward pass
        num_edges = features.shape[0]
        first = edges_topology[0, :] # first row
        second = edges_topology[1, :]  # second row

        # Execute T times
        for _ in range(self.hparams['T']):

            # We have the combination of the hidden states of the main edges with the neighbours
            mainEdges = tf.gather(features, first) # dim: (n_graphs x n_connections) x (hidden_dimension)
            neighEdges = tf.gather(features, second) # dim: (n_graphs x n_connections) x (hidden_dimension)

            edgesConcat = tf.concat([mainEdges, neighEdges], axis=1) # dim: (n_graphs x n_connections) x (2 x hidden_dimension)
 
            ### 1.a Message passing for link with all it's neighbours
            outputs = self.Message(edgesConcat) # dim: (n_graphs x n_connections) x (hidden_dimension)
            
            #print(outputs)
            #print(second)

            ### 1.b Sum of output values according to link id index
            edges_inputs = tf.math.unsorted_segment_sum(data=outputs, segment_ids=second, # dim: (n_graphs x n_edges) x (hidden_dimension)
                                                        num_segments=num_edges)
            
            ### 2. Update for each link
            # GRUcell needs a 3D tensor as state because there is a matmul: Wrap the link state
            outputs, links_state_list = self.Update(edges_inputs, [features]) # dim: (n_graphs x n_edges) x (hidden_dimension)
            
            link_state = links_state_list[0]

        # Perform sum of all hidden states
        edges_combi_outputs = tf.math.segment_sum(link_state, graph_ids, name=None)

        r = self.Readout(edges_combi_outputs,training=training)
        
        return r
