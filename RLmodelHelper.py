import tensorflow as tf
import numpy as np

from numpy.random import RandomState

## CHECKED AND SAFE
class PMF(tf.keras.Model):
    def __init__(self, n_users, n_items, n_dim):
        super(PMF, self).__init__()
        ## initializing attributes from parameters        
        self.w_u_i_init = tf.keras.initializers.RandomUniform(minval=-1., maxval=1., seed=1)
        
        ## initializing user embedding layer
        ## the output shape should be n_users * n_dim
        self.user_embedding = tf.keras.layers.Embedding(n_users,
                                                        n_dim,
                                                        embeddings_initializer='uniform',
                                                        embeddings_regularizer=tf.keras.regularizers.L2(0.1))
        ## initializing user embedding layer
        ## the output shape should be n_items * n_dim
        self.item_embedding = tf.keras.layers.Embedding(n_items,
                                                        n_dim,
                                                        embeddings_initializer='uniform',
                                                        embeddings_regularizer=tf.keras.regularizers.L2(0.1))
        
        ## users embedding
        self.ub = tf.keras.layers.Embedding(n_users, 
                                            1, 
                                            embeddings_initializer=self.w_u_i_init, 
                                            embeddings_regularizer=tf.keras.regularizers.L2(0.1))
        
        ## items embedding
        self.ib = tf.keras.layers.Embedding(n_items, 
                                            1, 
                                            embeddings_initializer=self.w_u_i_init, 
                                            embeddings_regularizer=tf.keras.regularizers.L2(0.1))
        
    def call(self, user_index, item_index):
        ## get the user and item embedding value
        user_h1 = self.user_embedding(user_index)
        item_h1 = self.item_embedding(item_index)
        ## should be checked again
        r_h = tf.math.reduce_sum(user_h1 * item_h1, axis=1 if len(user_h1.shape) > 1 else 0)
        r_h += tf.squeeze(self.ub(user_index))
        r_h += tf.squeeze(self.ib(item_index))
        return r_h

    def build_model(self):
        self.call(1, 1)
        return 'Weight Intialized'

## CHECKED SAVE
class DRRAveStateRepresentation(tf.keras.Model):
    def __init__(self, n_items=5, item_features=100, user_features=100):
        super(DRRAveStateRepresentation, self).__init__()
        ## initialize random_state to 1
        self.random_state = RandomState(1)
        
        ## hold all the parameters variable
        self.n_items = n_items
        self.item_features = item_features
        self.user_features = user_features
        
        ## add to the model parameter
        ## self.attention_weights shape (n_items x 1)
        ## later this need to be reshaped
        self.attention_weights = tf.Variable(initial_value=(0.1 * tf.random.uniform((n_items, 1), minval=0., maxval=1.)),
                                             trainable=True,
                                             dtype='float32')
        
    def call(self, user, items):
        '''
        items  : type(tensor) shape = (n_items x item_features)
        user   : type(tensor) shape = (user_features, )
        output : type(tensor) shape = (3 * item_features)
        '''
        ## right will result in numpy array
        ## because there is an issue with tensor matrix multiplications
        right = tf.transpose(items) @ self.attention_weights
        ## flatten the user
        right = tf.reshape(right, (right.shape[0],))
        middle = user * right
        output = tf.concat([user, middle, right], 0)
        output = tf.reshape(output, [1, output.shape[0]])
        return output
    
    def build_model(self):
        flow_item = tf.ones([self.n_items, self.item_features], tf.float32)
        flow_user = tf.ones([self.user_features,], tf.float32)
        self.call(flow_user, flow_item)
        return "Weight Initialized"

class Actor(tf.keras.Model):
    '''
    Actor network accounts for generatign action space based on
    the state space
    in_features : the size of state representation got from DRRAve
    out_features : the size of action space
    '''
    def __init__(self, in_features=100, out_features=18):
        super(Actor, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.linear_1 = tf.keras.layers.Dense(units=in_features,
                                              activation='relu',
                                              kernel_initializer=tf.keras.initializers.Orthogonal(seed=42),
                                              bias_initializer='zeros',
                                              kernel_regularizer=tf.keras.regularizers.L2(0.1))
        
        self.linear_2 = tf.keras.layers.Dense(units=in_features,
                                              activation='relu',
                                              kernel_initializer=tf.keras.initializers.Orthogonal(seed=42),
                                              bias_initializer='zeros',
                                              kernel_regularizer=tf.keras.regularizers.L2(0.1))
        
        self.linear_3 = tf.keras.layers.Dense(units=out_features,
                                              activation='tanh',
                                              kernel_initializer=tf.keras.initializers.Orthogonal(seed=42),
                                              bias_initializer='zeros',
                                              kernel_regularizer=tf.keras.regularizers.L2(0.1))
        
    def call(self, state):
        output = self.linear_1(state)
        output = self.linear_2(output)
        output = self.linear_3(output)
        return tf.convert_to_tensor(output)

    def build_model(self):
        flow_state = tf.ones([1, self.in_features], tf.float32)
        self.call(flow_state)
        return "Weight Initialized"

class Critic(tf.keras.Model):
    '''
    Critic networks are Deep-Q-Networks
    acton_size : is the size of action space from actor networks
    in_features : is the size of state representation got from DRR-Ave
    out_features : Q-Value
    '''
    def __init__(self, action_size=20, in_features=128, out_features=18):
        super(Critic, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.combo_features = in_features + action_size
        self.action_size = action_size
        ## check shape of the input
        self.linear_1 = tf.keras.layers.Dense(self.in_features, 
                                              activation='relu',
                                              kernel_initializer=tf.keras.initializers.Orthogonal(seed=42),
                                              bias_initializer='zeros',
                                              kernel_regularizer=tf.keras.regularizers.L2(0.1))
        
        self.linear_2 = tf.keras.layers.Dense(self.combo_features, 
                                              activation='relu',
                                              kernel_initializer=tf.keras.initializers.Orthogonal(seed=42),
                                              bias_initializer='zeros',
                                              kernel_regularizer=tf.keras.regularizers.L2(0.1))
        
        self.linear_3 = tf.keras.layers.Dense(self.combo_features, 
                                              activation='relu', 
                                              kernel_initializer=tf.keras.initializers.Orthogonal(seed=42),
                                              bias_initializer='zeros',
                                              kernel_regularizer=tf.keras.regularizers.L2(0.1))
        
        self.linear_4 = tf.keras.layers.Dense(out_features, 
                                              activation=None,
                                              kernel_initializer=tf.keras.initializers.Orthogonal(seed=42),
                                              bias_initializer='zeros',
                                              kernel_regularizer=tf.keras.regularizers.L2(0.1))
        
    def call(self, state, action):
        outputs = self.linear_1(state)
        outputs = self.linear_2(tf.concat([action, outputs], 1))
        outputs = self.linear_3(outputs)
        outputs = self.linear_4(outputs)
        return tf.convert_to_tensor(outputs)
    
    def build_model(self):
        flow_state = tf.ones([1, self.in_features], tf.float32)
        flow_action = tf.ones([1, self.action_size], tf.float32)
        self.call(flow_state, flow_action)
        return "Weight Initialized"