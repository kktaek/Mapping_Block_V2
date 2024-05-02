import numpy as np

import myutil as util

import tensorflow.keras as keras

#학습에서 사용하는 파라미터
class TrainParams:
    nEpochs = None
    nBatchSize_perClass = None
    model = None

    def __init__(self, ann_model, epochs, batch_size_per_class):
        self.nEpochs = epochs
        self.nBatchSize_perClass = batch_size_per_class
        self.model = ann_model


# 학습결과를 저장하는 클래스
class TrainResults:
    val_acc = None
    train_acc = None
    model = None


# train_batches_with_random_selection
# 주어진 데이터로부터 클래스별로 random selection 하여 학습하는 함수
# 현재 2진분류만 구현되어 있음
def train_batches_with_random_selection(train_parms: TrainParams, x_train, y_train, x_val=None, y_val=None, bVerbose=True) -> TrainResults:
    # 학습 파라미터 설정
    nEpochs = train_parms.nEpochs
    nBatchSize_perClass = train_parms.nBatchSize_perClass
    m = train_parms.model

    # target 값을 one-hot encoding 한다.
    y_train_cat = keras.utils.to_categorical(y_train)
    y_val_cat = keras.utils.to_categorical(y_val) if x_val is not None else 0

    # 결과 저장할 변수 생성
    val_acc = np.zeros(nEpochs) if x_val is not None else 0
    train_acc = np.zeros(nEpochs)

    # target 값에 따라 각각 index를 찾는다
    idx_random_zero = util.get_indices_where(y_train == 0)
    idx_random_one = util.get_indices_where(y_train == 1)
    idx_random_two = util.get_indices_where(y_train == 2)
    idx_random_three = util.get_indices_where(y_train == 3)
    idx_random_four = util.get_indices_where(y_train == 4)
    idx_random_five = util.get_indices_where(y_train == 5)
    idx_random_six = util.get_indices_where(y_train == 6)
    idx_random_seven = util.get_indices_where(y_train == 7)
    idx_random_eight = util.get_indices_where(y_train == 8)
    idx_random_nine = util.get_indices_where(y_train == 9)

    # 학습. nEpochs 수만큼 반복
    for i in range(nEpochs):
        # target 값이 0인 데이터와 1인 데이터의 인덱스를 각각 배치 수만큼 랜덤하게 선택한다.
        np.random.shuffle(idx_random_zero)
        np.random.shuffle(idx_random_one)
        np.random.shuffle(idx_random_two)
        np.random.shuffle(idx_random_three)
        np.random.shuffle(idx_random_four)
        np.random.shuffle(idx_random_five)
        np.random.shuffle(idx_random_six)
        np.random.shuffle(idx_random_seven)
        np.random.shuffle(idx_random_eight)
        np.random.shuffle(idx_random_nine)
        idx_random = np.concatenate([idx_random_zero[:nBatchSize_perClass], idx_random_one[:nBatchSize_perClass], idx_random_two[:nBatchSize_perClass],idx_random_three[:nBatchSize_perClass], idx_random_four[:nBatchSize_perClass], idx_random_five[:nBatchSize_perClass], idx_random_six[:nBatchSize_perClass], idx_random_seven[:nBatchSize_perClass], idx_random_eight[:nBatchSize_perClass], idx_random_nine[:nBatchSize_perClass]])

        # 인덱스를 사용하여 학습 데이터를 선택한다.
        x_train_batch = x_train[idx_random, :]
        y_train_batch = y_train_cat[idx_random, :]

        # 학습
        loss = m.train_on_batch(x_train_batch, y_train_batch)
        train_acc[i] = loss[1]
        if bVerbose:
            print(f'{i}/{nEpochs}: loss: {loss[0]} Acc: {loss[1]}', end='')

        # 검증
        if x_val is not None:
            loss_val = m.test_on_batch(x_val, y_val_cat)
            val_acc[i] = loss_val[1]
            if bVerbose:
                print(f'\t val_loss:{loss_val[0]} val_acc:{loss_val[1]}')
        else:
            if bVerbose:
                print('')

    out = TrainResults()
    out.train_acc = train_acc
    out.val_acc = val_acc
    out.model = m
    return out

# x,y 데이터를 주어진 비율에 맞추어 나누어 return
def split_data(x, y, test_ratio, val_ratio):
    n_row = x.shape[0]

    idx = np.array(range(n_row))
    np.random.shuffle(idx)  # 데이터 인덱스 섞기

    n_test = int(n_row * test_ratio)
    n_val = int(n_row * val_ratio)
    n_train = n_row - n_test - n_val

    idx_train = idx[:n_train]
    idx_val = idx[n_train:n_train + n_val]
    idx_test = idx[n_train + n_val:]

    x_train = x[idx_train, :]
    x_val = x[idx_val, :] if n_val > 0 else None
    x_test = x[idx_test, :]

    y_train = y[idx_train]
    y_val = y[idx_val] if n_val > 0 else None
    y_test = y[idx_test]

    return x_train, x_val, x_test, y_train, y_val, y_test


# N-Fold validation을 위한 데이터 분리
def split_data_for_N_fold_validation(x, y, fold=10, idx=0):
    n_row = x.shape[0]
    dx = n_row / fold
    s_id_test = int(np.round(dx * idx))
    e_id_test = min(int(np.round(dx * (idx+1))) - 1, n_row-1)

    # 각 클래스별 인덱스 추출
    class_indices = [np.where(y == i)[0] for i in range(10)]

    ids_test = []
    ids_train = []

    # 각 클래스별로 데이터를 균등하게 나누기
    for indices in class_indices:
        class_size = len(indices)
        s_id_class = int(np.round((class_size / n_row) * s_id_test))
        e_id_class = min(int(np.round((class_size / n_row) * (e_id_test + 1))) - 1, class_size - 1)

        ids_test.extend(indices[s_id_class:e_id_class + 1])
        ids_train.extend(np.concatenate([indices[:s_id_class], indices[e_id_class + 1:]]))

    ids_test = np.array(ids_test)
    ids_train = np.array(ids_train)

    # 데이터 섞기 (섞는 것이 중요할 수 있음)
    np.random.shuffle(ids_test)
    np.random.shuffle(ids_train)

    x_train = x[ids_train, :]
    x_test = x[ids_test, :]

    y_train = y[ids_train]
    y_test = y[ids_test]

    return x_train, x_test, y_train, y_test

