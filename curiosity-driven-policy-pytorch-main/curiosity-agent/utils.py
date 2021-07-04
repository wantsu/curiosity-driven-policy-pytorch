import torchvision.transforms as tfs
import scipy.signal


transform_train = tfs.Compose([tfs.ToPILImage(),
                           tfs.Resize([128, 128]),
                           tfs.RandomHorizontalFlip(),
                           tfs.ToTensor(),
                           tfs.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
                           ])

transform_val = tfs.Compose([tfs.ToPILImage(),
                           tfs.Resize([128, 128]),
                           tfs.ToTensor(),
                           tfs.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
                           ])

def discount(x, gamma):
    """
    compute discounted reward
        x = [r1, r2, r3, ..., rN]
        returns [r1 + r2*gamma + r3*gamma^2 + ...,
                   r2 + r3*gamma + r4*gamma^2 + ...,
                     r3 + r4*gamma + r5*gamma^2 + ...,
                        ..., ..., rN]
    """
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

