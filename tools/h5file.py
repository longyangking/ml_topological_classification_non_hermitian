import h5py

def save_data_M(M, filename):
    f = h5py.File(filename, 'w')
    f.create_dataset('data', data=M)
    f.close()

def load_data_M(filename):
    f = h5py.File(filename, 'r')
    M = f['data'][:]
    f.close()
    return M

def save_data(Xs, ys, filename):
    f = h5py.File(filename, 'w')
    f.create_dataset('Xs', data=Xs)
    f.create_dataset('ys', data=ys)
    f.close()

def load_data(filename):
    f = h5py.File(filename, 'r')
    Xs = f['Xs'][:]
    ys = f['ys'][:]
    f.close()
    return Xs, ys