#myutil.py
#
# by prof. Won-Du Chang (12cross@gmail.com)
#

import numpy as np


# get_indices_where
# numpy 배열 x 에서 condition==True 을 만족하는 값의 indices 리턴한다.
# @param: condition (boolean 배열. x와 배열의 길이(모양)가 같아야 한다.)
# @param: x (1차원 배열이어야 함)
#
# return: 조건을 만족하는 값들의 위치(인덱스). numpy 배열 형태
def get_indices_where(condition: np.ndarray) -> np.ndarray:
    indices = np.zeros(condition.shape[0],dtype=int)
    cnt = 0
    for i in range(condition.shape[0]):
        if condition[i]:
            indices[cnt] = i
            cnt += 1

    indices = indices[:cnt]
    return indices


