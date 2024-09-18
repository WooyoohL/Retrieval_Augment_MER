import faiss
import numpy as np
import os
import pandas as pd
import torch
import tqdm


def normalize_to_probability(X):
    return X / np.sum(X, axis=1, keepdims=True)


def get_use_sample_name(label_path, trainset=None):
    samples = []
    if label_path.endswith('.npz'):
        label_file = np.load(label_path, allow_pickle=True)
        # label = label_file['train_corpus'].item()
        if trainset == 1:
            label = label_file['train_corpus'].item()
        elif trainset == 0:
            label = label_file['test1_corpus'].item()

        for i, sample in enumerate(label):
            samples.append(sample)
    elif label_path.endswith('.pkl'):
        label_file = pd.read_pickle(label_path)
        for i, sample in enumerate(label_file):
            samples.append(sample)

    return samples


def get_sample_path(samples, audio_data_path, video_data_path, text_data_path):
    # 读取所有feature
    feature_files_modality_A = [os.path.join(audio_data_path, f + '.npy') for f in samples]
    feature_files_modality_V = [os.path.join(video_data_path, f + '.npy') for f in samples]
    feature_files_modality_L = [os.path.join(text_data_path, f + '.npy') for f in samples]
    return feature_files_modality_A, feature_files_modality_V, feature_files_modality_L


#
def load_features_mer(files):
    features_list = []
    for file in tqdm.tqdm(files):
        features = np.load(file).astype('float32')  # 确保特征是float32类型
        features_list.append(features)
    features_list = np.stack(features_list, axis=0)
    # print(features_list.shape)
    return features_list#.squeeze(axis=1)



def load_all_feats(feature_files_modality_A, feature_files_modality_V, feature_files_modality_L, dataset):
    if dataset == 'MER2024':
        features_modality1 = load_features_mer(feature_files_modality_A)
        # print(features_modality1.shape)
        features_modality2 = load_features_mer(feature_files_modality_V)
        features_modality3 = load_features_mer(feature_files_modality_L)
    return features_modality1, features_modality2, features_modality3


# # #
def save_fassi_vector_store(save_path, features_modality1, features_modality2, features_modality3):

    print(features_modality1[0])
    d1 = features_modality1.shape[1]
    d2 = features_modality2.shape[1]
    d3 = features_modality3.shape[1]
    print(d1, d2, d3)
    #

    faiss.normalize_L2(features_modality1)
    faiss.normalize_L2(features_modality2)
    faiss.normalize_L2(features_modality3)

    # create index
    index1 = faiss.IndexFlatIP(d1)
    index2 = faiss.IndexFlatIP(d2)
    index3 = faiss.IndexFlatIP(d3)


    index1.add(features_modality1)
    index2.add(features_modality2)
    index3.add(features_modality3)

    # 保存索引到磁盘
    faiss.write_index(index1, os.path.join(save_path, 'faiss_index_modality_A.index'))
    faiss.write_index(index2, os.path.join(save_path, 'faiss_index_modality_V.index'))
    faiss.write_index(index3, os.path.join(save_path, 'faiss_index_modality_L.index'))
    print("FAISS index is successfully created and saved.")



def read_indexs(index_A_path, index_V_path, index_L_path):
    index_A = faiss.read_index(index_A_path)
    index_V = faiss.read_index(index_V_path)
    index_L = faiss.read_index(index_L_path)
    return index_A, index_V, index_L



class Retrieval_augment:
    def __init__(self, dataset, k, datascale=None):
        if datascale is None:
            # datascale = ['MER_UNI', 'MER_UNI_11W', 'MER_UNI_ALL']
            datascale = 'MER_UNI'
        if dataset == 'MER2024':
            # feature/indexsave/label path
            save_path = '/MER24_FAISS_Vector_Store/save/'
            hidden_text_data_path = f'./MER2024/{datascale}/Baichuan2-13B-Base-UTT'
            hidden_audio_data_path = f'./MER2024/{datascale}/chinese-hubert-large-UTT'
            hidden_video_data_path = f'./MER2024/{datascale}/clip-vit-large-patch14-UTT'
            label_path = './MER2024/mer2024-dataset-process/label-6way.npz'
            # index path
            index_A_path = f'./MER24_FAISS_Vector_Store/{datascale}/faiss_index_modality_A.index'
            index_V_path = f'/./MER24_FAISS_Vector_Store/{datascale}/faiss_index_modality_V.index'
            index_L_path = f'./MER24_FAISS_Vector_Store/{datascale}/faiss_index_modality_L.index'

            if datascale == 'MER_UNI':
                self.samples = get_use_sample_name(label_path, trainset=1)
            elif datascale == 'MER_UNI_11W':
                self.samples = get_use_sample_name(label_path, trainset=0)
            elif datascale == 'MER_UNI_ALL':
                self.samples = get_use_sample_name(label_path, trainset=0)
                self.samples += get_use_sample_name(label_path, trainset=1)

        feature_files_modality_A_path, feature_files_modality_V_path, \
        feature_files_modality_L_path = get_sample_path(self.samples, hidden_audio_data_path, hidden_video_data_path, hidden_text_data_path)

        self.feat_A, self.feat_V, self.feat_L = load_all_feats(feature_files_modality_A_path,
                                                               feature_files_modality_V_path,
                                                               feature_files_modality_L_path, dataset)

        self.index_A, self.index_V, self.index_L = read_indexs(index_A_path, index_V_path, index_L_path)
        self.d_A = self.index_A.d  # the embd_size of the faiss index
        self.d_V = self.index_V.d
        self.d_L = self.index_L.d
        self.k = k


    def get_most_similar_vectors(self, query_vector1, query_vector2=None, search_type=None):
        debug = False
        query_vector1 = np.expand_dims(query_vector1.detach().cpu(), axis=0)
        if query_vector2 is not None:
            query_vector2 = np.expand_dims(query_vector2.detach().cpu(), axis=0)

        most_similar_vectors = []
        origin_similar_vectors = []
        if search_type in ['A', 'V', 'L']:
            index = getattr(self, f'index_{search_type}')
            distances, indices = index.search(query_vector1, self.k)
            # print(indices)
            if debug:
                print(f'D{search_type}:', distances)
            indices = indices[:, 1:]
            most_similar_indices = indices[0]

            other_feats = {'A': ['feat_A', 'feat_V', 'feat_L'],
                           'V': ['feat_A', 'feat_V', 'feat_L'],
                           'L': ['feat_A', 'feat_V', 'feat_L']}[search_type]

            most_similar_vectors.append([getattr(self, other_feats[0])[idx] for idx in most_similar_indices])
            most_similar_vectors.append([getattr(self, other_feats[1])[idx] for idx in most_similar_indices])
            most_similar_vectors.append([getattr(self, other_feats[2])[idx] for idx in most_similar_indices])
            distances = distances[0]
        elif search_type in ['AV', 'AL', 'VL']:
            type1, type2 = search_type
            index1 = getattr(self, f'index_{type1}')
            index2 = getattr(self, f'index_{type2}')
            distances1, indices1 = index1.search(query_vector1, self.k)
            distances2, indices2 = index2.search(query_vector2, self.k)

            if debug:
                print(f'D{search_type}:', distances1, distances2)
            indices = indices1[:, 1:] if distances1[0][-1] > distances2[0][-1] else indices2[:, 1:]

            distances = distances1[0] if distances1[0][-1] > distances2[0][-1] else distances2[0]

            most_similar_indices = indices[0]

            most_similar_vectors.append([self.feat_A[idx] for idx in most_similar_indices])
            most_similar_vectors.append([self.feat_V[idx] for idx in most_similar_indices])
            most_similar_vectors.append([self.feat_L[idx] for idx in most_similar_indices])

        return most_similar_vectors, origin_similar_vectors, [self.samples[idx] for idx in most_similar_indices], distances, search_type






if __name__ == '__main__':
    # here is the step to create the faiss index
    dataset = 'MER2024'

    if dataset == 'MER2024':
        # feature/indexsave/label path
        save_path = './MER24_FAISS_Vector_Store/MER_UNI'
        # here is the emotion hidden feature save path, you need to inference the dataset and get them
        text_data_path = './MER2024/MER_UNI/Baichuan2-13B-Base-UTT'
        audio_data_path = './MER2024/MER_UNI/chinese-hubert-large-UTT'
        video_data_path = './MER2024/MER_UNI/clip-vit-large-patch14-UTT'
        label_path = '/workspace/fanqi/dataset/MER2024/mer2024-dataset-process/label-6way.npz'

        index_A_path = './MER24_FAISS_Vector_Store/MER_UNI/faiss_index_modality_A.index'
        index_V_path = './MER24_FAISS_Vector_Store/MER_UNI/faiss_index_modality_V.index'
        index_L_path = './MER24_FAISS_Vector_Store/MER_UNI/faiss_index_modality_L.index'
    # # save index
    samples = get_use_sample_name(label_path, trainset=1)
    # samples += get_use_sample(label_path, trainset=0)

    feature_files_modality_A_path, feature_files_modality_V_path, \
    feature_files_modality_L_path = get_sample_path(samples, audio_data_path, video_data_path, text_data_path)
    #
    features_modality1, features_modality2, features_modality3 = load_all_feats(feature_files_modality_A_path,
                                                                                feature_files_modality_V_path,
                                                                                feature_files_modality_L_path, dataset)
    print(features_modality1.shape)
    save = True
    if save:
        save_fassi_vector_store(save_path, features_modality1, features_modality2, features_modality3)
    # #、
    # read the index
    # index_A, index_V, index_L = read_indexs(index_A_path, index_V_path, index_L_path)

