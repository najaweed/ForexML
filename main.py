import numpy as np
import itertools


def ordinal_sequence(data, dx=3, dy=1, taux=1, tauy=1, overlapping: bool = False, ):
    try:
        ny, nx = np.shape(data)
        data = np.array(data)
    except:
        nx = np.shape(data)[0]
        ny = 1
        data = np.array([data])

    # TIME SERIES DATA
    if ny == 1:
        if overlapping:
            partitions = np.concatenate(
                [
                    [np.concatenate(data[j:j + dy * tauy:tauy, i:i + dx * taux:taux]) for i in
                     range(nx - (dx - 1) * taux)]
                    for j in range(ny - (dy - 1) * tauy)
                ]
            )

        else:  # non overlapping
            partitions = np.concatenate(
                [
                    [np.concatenate(data[j:j + dy * tauy:tauy, i:i + dx * taux:taux]) for i in
                     range(0, nx - (dx - 1) * taux, dx + (dx - 1) * (taux - 1))]
                    for j in range(ny - (dy - 1) * tauy)
                ]
            )
        #
        symbols = np.apply_along_axis(np.argsort, 1, partitions)

    # IMAGE DATA
    else:
        if overlapping:
            partitions = np.concatenate(
                [
                    [[np.concatenate(data[j:j + dy * tauy:tauy, i:i + dx * taux:taux]) for i in
                      range(nx - (dx - 1) * taux)]]
                    for j in range(ny - (dy - 1) * tauy)
                ]
            )

        else:  # non overlapping
            partitions = np.concatenate(
                [
                    [[np.concatenate(data[j:j + dy * tauy:tauy, i:i + dx * taux:taux]) for i in
                      range(0, nx - (dx - 1) * taux, dx + (dx - 1) * (taux - 1))]]
                    for j in range(0, ny - (dy - 1) * tauy, dy + (dy - 1) * (tauy - 1))
                ]
            )
            #
        symbols = np.apply_along_axis(np.argsort, 2, partitions)

    return symbols


def _embed(x, order=3, delay=1):
    """Time-delay embedding.
    Parameters
    ----------
    x : 1D-array of shape (n_times) or 2D-array of shape (signal_indice, n_times)
    order : int
        Embedding dimension (order).
    delay : int
        Delay.
    Returns
    -------
    embedded : 2D-array (if x is 1D)
        Embedded time-series, of shape (n_times - (order - 1) * delay, order)
    embedded : 3D-array if (x is 2D)
        Embedded time-series, of shape (signal_indice, n_times - (order - 1) * delay, order_num)
    """

    assert type(order) == int, "order must be integer!"
    # check order is int

    N = x.shape[-1]
    if order * delay > N:
        raise ValueError("Error: order * delay should be lower than x.size")
    if delay < 1:
        raise ValueError("Delay has to be at least 1.")
    if order < 2:
        raise ValueError("Order has to be at least 2.")
    # check parameters

    if x.ndim == 1:
        # pass 1D array

        Y = np.zeros((order, N - (order - 1) * delay))
        for i in range(order):
            Y[i] = x[(i * delay):(i * delay + Y.shape[1])]
        return Y.T

    else:
        # pass 2D array

        Y = []
        # pre-defiend an empty list to store numpy.array (concatenate with a list is faster)

        embed_signal_length = N - (order - 1) * delay
        # define the new signal length

        indice = [[(i * delay), (i * delay + embed_signal_length)] for i in range(order)]
        # generate a list of slice indice on input signal

        for i in range(order):
            # loop with the order
            temp = x[:, indice[i][0]: indice[i][1]].reshape(-1, embed_signal_length, 1)
            # slicing the signal with the indice of each order (vectorized operation)

            Y.append(temp)
            # append the sliced signal to list

        Y = np.concatenate(Y, axis=-1)
        # concatenate the sliced signal to a 3D array (signal_indice, n_times - (order - 1) * delay, order_num)
        return Y


def pec(y, D, t):
    y_len = len(y)
    serial = np.arange(0, D)
    y_perm = list(itertools.permutations(serial, D))
    count = np.zeros(len(y_perm))
    '''for item in y_perm:
        print(item)
    '''
    embd = []
    for i in range(y_len - (D - 1) * t):
        y_x = np.argsort(y[i:i + t * D:t])
        # print(tuple(y_x))
        embd.append(y_x)
        for j in range(len(y_perm)):
            if tuple(y_x) == y_perm[j]:
                count[j] += 1

    # plt.hist(count)
    # plt.show()

    # pe = scipy.stats.entropy(count / (y_len-(D-1)*t), base=2)
    # print(pe)
    return count, embd, y_perm


a = np.arange(0, 11)
np.random.shuffle(a)
np.random.shuffle(a)

b = np.arange(00, 11)
np.random.shuffle(b)
c = np.arange(00, 11)
np.random.shuffle(c)
x = np.array([a, b, c, c])

# print(a)
# print(pec(a,3,1))

ord = _embed(x, order=4, delay=1)  # ordinal_sequence(a,dx=3,taux=3)
print(ord, ord.shape)
permt = np.apply_along_axis(np.argsort, 2, ord)
print(permt.argsort(kind='quicksort'))


# sorted_idx = permt.argsort(kind='quicksort')
# print(np.unique(sorted_idx, return_counts=True, axis=1))


def ordinal_distribution(data, dx=3, dy=1, taux=1, tauy=1, return_missing=False, tie_precision=None):
    def setdiff(a, b):
        """
        Searches for elements (subarrays) in `a` that are not contained in `b` [*]_.
        Parameters
        ----------
        a : tuples, lists or arrays
            Array in the format :math:`[[x_{21}, x_{22}, x_{23}, \\ldots, x_{2m}],
            \\ldots, [x_{n1}, x_{n2}, x_{n3}, ..., x_{nm}]]`.
        b : tuples, lists or arrays
            Array in the format :math:`[[x_{21}, x_{22}, x_{23}, \\ldots, x_{2m}],
            \\ldots, [x_{n1}, x_{n2}, x_{n3}, ..., x_{nm}]]`.

        Returns
        -------
        : array
            An array containing the elements in `a` that are not contained in `b`.
        Notes
        -----
        .. [*] This function was adapted from https://stackoverflow.com/questions/8317022/get-intersecting-rows-across-two-2d-numpy-arrays
        Examples
        --------
        >>> a = ((0,1,2), (0,1,2), (1,0,2), (2,0,1))
        >>> b = [[0,2,1], [0,1,2], [0,1,2]]
        >>> setdiff(a, b)
        array([[1, 0, 2],
            [2, 0, 1]])
        """
        a = np.asarray(a).astype('int64')
        b = np.asarray(b).astype('int64')

        _, ncols = a.shape

        dtype = {'names': ['f{}'.format(i) for i in range(ncols)],
                 'formats': ncols * [a.dtype]}

        C = np.setdiff1d(a.view(dtype), b.view(dtype))
        C = C.view(a.dtype).reshape(-1, ncols)

        return (C)

    #####################################################################################

    try:
        ny, nx = np.shape(data)
        data = np.array(data)
    except:
        nx = np.shape(data)[0]
        ny = 1
        data = np.array([data])

    if tie_precision is not None:
        data = np.round(data, tie_precision)

    partitions = np.concatenate(
        [
            [np.concatenate(data[j:j + dy * tauy:tauy, i:i + dx * taux:taux]) for i in range(nx - (dx - 1) * taux)]
            for j in range(ny - (dy - 1) * tauy)
        ]
    )

    symbols = np.apply_along_axis(np.argsort, 1, partitions)
    symbols, symbols_count = np.unique(symbols, return_counts=True, axis=0)

    probabilities = symbols_count / len(partitions)

    if return_missing == False:
        return symbols, probabilities

    else:
        all_symbols = list(map(list, list(itertools.permutations(np.arange(dx * dy)))))
        miss_symbols = setdiff(all_symbols, symbols)
        # symbols = np.concatenate((symbols, miss_symbols))
        probabilities = np.concatenate((probabilities, np.zeros(miss_symbols.__len__())))

        return symbols, miss_symbols, probabilities


def mpe(mts, m, d):
    """
    This function attempts to compute the multivariate permutation entropy of a multivariate time series.
    The algorithm follows the formulations of Morabito et al. (2012) and it is based for the most part on
    Nikolay Donets' pyEntropy code (2012).

    INPUTS are:
        - mts: multivariate time series of shape "N series x N samples"
        - m: order of possible permutation motifs
        - d: time-lag

    OUTPUTS are:
        - pe_channel: single series entropy
        - pe_cross: cross-series entropy

    References:
        - Morabito, F.C., Labate, D., La Foresta, F., Bramanti, A., Morabito, G. & Palamara, I. (2012).
          Multivariate Multi-Scale Permutation Entropy for Complexity Analysis of Alzheimer’s Disease EEG.
          Entropy, 14, 1186-1202.
        - Donets, N. (2013). PyEntropy. Github repository, https://github.com/nikdon/pyEntropy
    """
    # initialize parameters
    n = len(mts[0])
    e = len(mts)
    permutations = np.array(list(itertools.permutations(range(m))))
    t = n - d * (m - 1)
    c = []
    p = []
    pe_channel = []

    for j in range(e):
        c.append([0] * len(permutations))

    # compute single series permutation entropy based on the multivariate distribution of motifs
    for f in range(e):
        for i in range(t):
            sorted_index_array = np.array(np.argsort(mts[f][i:i + d * m:d], kind='quicksort'))
            for j in range(len(permutations)):
                if abs(permutations[j] - sorted_index_array).any() == 0:
                    c[f][j] += 1

        p.append(np.divide(np.array(c[f]), float(t * e)))
        pe_channel.append(-np.nansum(p[f] * np.log2(p[f])))

    # compute the cross-series permutation entropy based on the multivariate distribution of motifs
    rp = []
    pe_cross = []
    for w in range(len(permutations)):
        rp.append(np.nansum(np.array(p)[:, w]))

    pe_cross = -np.nansum(rp * np.log2(rp))

    return pe_channel, pe_cross
# print(mpe(x.T,2,1))
# print(a)
# permt,miss_permt , dist = ordinal_distribution(a,dx=4,dy=1,return_missing=True)
# print(list(map(list, list(itertools.permutations(np.arange(4))))))
# symbols, symbols_count = np.unique(symbols, return_counts=True, axis=0)
# print(ordinal_distribution(a,dx=4,dy=1,return_missing=True))
