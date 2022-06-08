import pickle
import numpy as np
import tensorflow as tf

from RLmodelHelper import Actor, Critic, DRRAveStateRepresentation, PMF
from RLutils.history_buffer import HistoryBuffer


class Recommender():
    def __init__(self):
        self.path = "models/Deep Reinforcement Learning (DRR)/"
        self.users = pickle.load(open(self.path+'dataset_RL/user_id_to_num.pkl', 'rb'))
        self.items = pickle.load(open(self.path+'dataset_RL/item_lookup.pkl', 'rb'))
        self.data = np.load(self.path+'dataset_RL/data_RL_25000.npy')
        self.data[:, 2] = 0.5 * (self.data[:, 2] - 3)

        ## deklarasi model
        self.state_rep_net = DRRAveStateRepresentation(5, 100, 100)
        self.actor_net = Actor(300, 100)
        self.critic_net = Critic(100, 300, 1)
        self.reward_function = PMF(len(self.users), len(self.items), 100)

        ## deklarasi build
        self.state_rep_net.build_model()
        self.actor_net.build_model()
        self.critic_net.build_model()
        self.reward_function.build_model()

        ## load data
        self.train_data = tf.convert_to_tensor(self.data[:int(0.8 * self.data.shape[0])], dtype='float32')

        ## sifat user dan item
        self.user_embeddings = tf.convert_to_tensor(self.reward_function.user_embedding.get_weights()[0])
        self.item_embeddings = tf.convert_to_tensor(self.reward_function.item_embedding.get_weights()[0])
        self.candidate_items = tf.identity(tf.stop_gradient(self.item_embeddings))

    def getId(self, dest):
        x=0
        for i in self.items:
            if self.items.get(x).lower() == dest.lower():
                return x
            x+=1
        return 0
    
    def loadParam(self):
        self.reward_function.load_weights(self.path+'trained/pmf_weights/pmf_150')
        self.actor_net.load_weights(self.path+'trained/actor_weights/actor_150')
        self.critic_net.load_weights(self.path+'trained/critic_weights/critic_150')
        self.state_rep_net.load_weights(self.path+'trained/state_rep_weights/state_rep_150')

    def recommend(self,data):

        history_buffer = HistoryBuffer(5)
        e = np.random.randint(0, len(self.users))

        user_reviews = self.train_data[self.train_data[:, 0] == e]
        pos_user_reviews = user_reviews[user_reviews[:, 2] > 0]

        if pos_user_reviews.shape[0] < 5:
            while pos_user_reviews.shape[0] < 5:
                e = np.random.randint(0, len(self.users))
                user_reviews = self.train_data[self.train_data[:, 0] == e]
                pos_user_reviews = user_reviews[user_reviews[:, 2] > 0]

        ## copy embedding item
        candidate_items = tf.identity(tf.stop_gradient(self.item_embeddings))

        int_user_reviews_idx = user_reviews[:, 1].numpy().astype(int)
        user_candidate_items = tf.identity(tf.stop_gradient(tf.gather(self.item_embeddings, indices=int_user_reviews_idx)))

        user_emb = self.user_embeddings[e]

        ignored_items = []
        for i in data:
            idx = self.getId(i.destination)
            if i.like > 0:
                emb = candidate_items[idx]
                history_buffer.push(emb)
                ignored_items.append(idx)
                
        if len(history_buffer) < 5:
            i = 0
            while len(history_buffer) < 5:
                emb = candidate_items[int(pos_user_reviews[i, 1].numpy())]
                history_buffer.push(tf.identity(tf.stop_gradient(emb)))
                i+=1

        t = 0
        explore = 0.2
        state = None
        action = None
        reward = None
        next_state = None
        recommend_item = []
        user_ignored_items = []

        self.loadParam()
        while t < 10:
            state = self.state_rep_net(user_emb, tf.stack(history_buffer.to_list()), training=False)
            if np.random.uniform(0, 1) < explore:
                action = tf.convert_to_tensor(0.1 * np.random.rand(1,100), dtype='float32')
            else:
                action = self.actor_net(state, training=False)
                
            ranking_scores = candidate_items @ tf.transpose(action)
            ranking_scores = tf.reshape(ranking_scores, (ranking_scores.shape[0],)).numpy()
            
            if len(ignored_items) > 0:
                rec_items = tf.stack(ignored_items).numpy().astype(int)
                rec_items = np.array(rec_items)
            else:
                rec_items = []
            
            user_rec_items = tf.stack(user_ignored_items) if len(user_ignored_items) > 0 else []
            user_rec_items = np.array(user_rec_items)
            
            ranking_scores[rec_items if len(ignored_items) > 0 else []] = -float("inf")
            user_ranking_scores = tf.gather(ranking_scores, indices=user_reviews[:, 1].numpy().astype(int)).numpy()
            user_ranking_scores[user_rec_items if len(user_ignored_items) > 0 else []] = -float("inf")
            
            rec_item_idx = tf.math.argmax(user_ranking_scores).numpy()    
            rec_item_emb = user_candidate_items[rec_item_idx]
            
            recommend_item.append(self.items.get(rec_item_idx))
            reward = self.reward_function(float(e), float(rec_item_idx))
            if reward > 0:
                history_buffer.push(tf.identity(tf.stop_gradient(rec_item_emb)))
                next_state = self.state_rep_net(user_emb, tf.stack(history_buffer.to_list()))
            else:
                next_state = tf.stop_gradient(state)
            
            user_ignored_items.append(rec_item_idx)
            t+=1

        return recommend_item