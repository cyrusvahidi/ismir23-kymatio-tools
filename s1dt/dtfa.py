import tqdm
import torch
import torch.nn as nn

from kymatio.torch import TimeFrequencyScattering
from plot import mesh_plot_3d, plot_contour_gradient

class DistanceLoss(nn.Module):
    """
    The DistanceLoss class is a PyTorch module that computes the distance between two tensors using a specified p-norm.

    Attributes:
        p (float): The p-norm to use for computing the distance.

    Methods:
        create_ops(*args, **kwargs): This method must be implemented in the subclass and should create the operators to apply to the input tensors.
        dist(x, y): Computes the distance between two tensors using the specified p-norm.
        forward(x, y, transform_y=True): Computes the distance loss between two tensors.

    Raises:
        NotImplementedError: Raised if the create_ops method is not implemented in the subclass.
    """
    def __init__(self, p=2):
        """
        Initializes a new instance of the DistanceLoss class.

        Args:
            p (float): The p-norm to use for computing the distance.
        """
        super().__init__()
        self.p = p

    def create_ops(*args, **kwargs):
        """
        This method must be implemented in the subclass and should create the operators to apply to the input tensors.

        Raises:
            NotImplementedError: This method must be implemented in the subclass.
        """
        raise NotImplementedError

    def dist(self, x, y):
        """
        Computes the distance between two tensors using the specified p-norm.

        Args:
            x (Tensor): The first tensor.
            y (Tensor): The second tensor.

        Returns:
            Tensor: The distance between the two tensors.
        """
        if self.p == 1.0:
            return torch.abs(x - y).mean()
        elif self.p == 2.0:
            return torch.norm(x - y, p=2.0)

    def forward(self, x, y, transform_y=True):
        """
        Computes the distance loss between two tensors.

        Args:
            x (Tensor): The first tensor.
            y (Tensor): The second tensor.
            transform_y (bool): Whether to apply the operators to the second tensor.

        Returns:
            Tensor: The distance loss between the two tensors.

        Raises:
            NotImplementedError: Raised if the create_ops method is not implemented in the subclass.
        """
        loss = torch.tensor(0.0).type_as(x)
        for op in self.ops:
            loss += self.dist(op(x), op(y) if transform_y else y)
        loss /= len(self.ops)
        return loss


class TimeFrequencyScatteringLoss(DistanceLoss):
    """
    The TimeFrequencyScatteringLoss class is a subclass of the DistanceLoss class that computes the distance between two tensors using the Time-Frequency scattering transform.

    Attributes:
        shape (tuple): The shape of the input tensors.
        Q (tuple): The Q-factor and J-factor to use for the Time-Frequency scattering transform.

    Methods:
        create_ops(x, y): Creates the Time-Frequency scattering operators to apply to the input tensors.
    """
    def __init__(
        self,
        shape,
        Q=(8, 1),
        J=12,
        J_fr=3,
        Q_fr=2,
        F=None,
        T=None,
        format="time",
        p=2.0,
    ):
        """
        Initializes a new instance of the TimeFrequencyScatteringLoss class.

        Args:
            shape (tuple): The shape of the input tensors.
            Q (tuple): The Q-factor and J-factor to use for the Time-Frequency scattering transform.
        """
        super().__init__(p=p)

        self.shape = shape
        self.Q = Q
        self.J = J
        self.J_fr = J_fr
        self.F = F
        self.Q_fr = Q_fr
        self.T = T
        self.format = format
        self.create_ops()

    def create_ops(self):
        """
        Creates the Time-Frequency scattering operators to apply to the input tensors.

        Args:
            x (Tensor): The first tensor.
            y (Tensor): The second tensor.

        Returns:
            list: The list of Time-Frequency scattering operators to apply to the input tensors.
        """
        S = TimeFrequencyScattering(
            shape=self.shape,
            Q=self.Q,
            J=self.J,
            J_fr=self.J_fr,
            Q_fr=self.Q_fr,
            T=self.T,
            F=self.F,
            format=self.format,
        ).cuda()
        self.ops = [S]


class MultiScaleSpectralLoss(DistanceLoss):
    """
    Multi-scale spectral loss module.

    Args:
        max_n_fft (int, optional): The maximum size of the FFT (Fast Fourier Transform). Defaults to 2048.
        num_scales (int, optional): The number of scales to consider. Defaults to 6.
        hop_lengths (list, optional): The hop lengths for each scale. If not provided, they are computed automatically. Defaults to None.
        mag_w (float, optional): The weight for the magnitude component. Defaults to 1.0.
        logmag_w (float, optional): The weight for the log-magnitude component. Defaults to 0.0.
        p (float, optional): The exponent value for the distance metric. Defaults to 1.0.

    Notes:
        - The `max_n_fft` parameter should be divisible by 2 raised to the power of (`num_scales` - 1).
        - If `hop_lengths` are not provided, they are automatically computed based on the `n_ffts` of each scale.

    Example:
        >>> loss = MultiScaleSpectralLoss(max_n_fft=4096, num_scales=4, mag_w=0.8, logmag_w=0.2, p=2.0)
    """
    def __init__(
        self,
        max_n_fft=2048,
        num_scales=6,
        hop_lengths=None,
        mag_w=1.0,
        logmag_w=0.0,
        p=1.0,
    ):
        super().__init__(p=p)
        assert max_n_fft // 2 ** (num_scales - 1) > 1
        self.max_n_fft = 2048
        self.n_ffts = [max_n_fft // (2**i) for i in range(num_scales)]
        self.hop_lengths = (
            [n // 4 for n in self.n_ffts] if not hop_lengths else hop_lengths
        )
        self.mag_w = mag_w
        self.logmag_w = logmag_w

        self.create_ops()

    def create_ops(self):
        self.ops = [
            MagnitudeSTFT(n_fft, self.hop_lengths[i])
            for i, n_fft in enumerate(self.n_ffts)
        ]


class MagnitudeSTFT(nn.Module):
    """
    The MagnitudeSTFT class is a PyTorch module that computes the magnitude of the Short-Time Fourier Transform (STFT) of a tensor.

    Attributes:
        n_fft (int): The number of FFT points to use for computing the STFT.
        hop_length (int): The number of samples to advance between STFT frames.
        win_length (int): The length of the STFT window.
        window (Tensor): The window function to use for the STFT.
        center (bool): Whether to pad the input tensor before computing the STFT.
        normalized (bool): Whether to normalize the STFT by the window energy.
        onesided (bool): Whether to return only the positive frequencies of the STFT.

    Methods:
        forward(x): Computes the magnitude of the STFT of a tensor.

    Raises:
        ValueError: Raised if the window tensor is not 1-dimensional.
    """
    def __init__(self, n_fft, hop_length):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length

    def forward(self, x):
        """
        Computes the magnitude of the STFT of a tensor.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The magnitude of the STFT of the input tensor.
        """
        return torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=torch.hann_window(self.n_fft).type_as(x),
            return_complex=True,
        ).abs()


# class DTFA:

#     def __init__(self, dist="jtfs", am_range=(4, 16), fm_range=(0.5, 4), N=20, target=None, f0=768.0, sr=2**12, time_shift=None):
#         self.thetas = thetas
#         self.dist = dist
        
#         self.f0 = f0
#         self.sr = sr
#         self.N = N

#         self.target = target
#         self.time_shift = time_shift
#         self.target_idx = N * (N // 2) + (N // 2)

#         self.AM, self.FM = grid2d(x1=am_range[0], x2=am_range1, y1=fm_range[0], y2=fm_range[1], n=N)
#         self.AM.requires_grad = True
#         self.FM.requires_grad = True
#         self.thetas = torch.stack([self.AM, self.FM], dim=-1).cuda()

#         theta_target = thetas[target_idx].clone().detach().requires_grad_(False)
#         self.target = (
#             generate_am_chirp(
#                 [f0, theta_target[0], theta_target[1]], sr=sr, duration=duration
#             )
#             .cuda()
#             .detach()
#         )

#         if dist == "jtfs":
#             loss_fn = TimeFrequencyScatteringLoss(
#                 shape=(sr * duration,),
#                 Q=(8, 2),
#                 J=12,
#                 J_fr=5,
#                 F=0,
#                 Q_fr=2,
#                 format="time",
#             )
#             Sx_target = loss_fn.ops[0](target.cuda()).detach()
#         elif dist == "mss":
#             loss_fn = MultiScaleSpectralLoss(max_n_fft=1024)

#     def compute_grads(self, target):
#         losses, grads = [], []
#         for theta in tqdm.tqdm(self.thetas):
#             am = torch.tensor(theta[0], requires_grad=True, dtype=torch.float32)
#             fm = torch.tensor(theta[1], requires_grad=True, dtype=torch.float32)
#             audio = generate_am_chirp(
#                 [torch.tensor([self.f0, dtype=torch.float32, requires_grad=False).cuda(), am, fm],
#                 sr=sr,
#                 duration=duration,
#                 delta=(2 ** random.randint(8, 12) if self.time_shift == "random" else 2**8)
#                 if self.time_shift
#                 else 0,
#             )

#             loss = (
#                 loss_fn(audio.cuda(), self.target.cuda(), transform_y=False)
#                 if self.target
#                 else loss_fn(audio, target)
#             )
#             loss.backward()
#             losses.append(float(loss.detach().cpu().numpy()))
#             x.append(float(am))
#             y.append(float(fm))
#             u.append(float(-am.grad))
#             v.append(float(-fm.grad))

#             grad = np.stack([float(-am.grad), float(-fm.grad)])
#             grads.append(grad)
#         self.grads = grads 
#         self.losses = losses 
    
#     def meshplot(self):
#         X = self.AM.numpy().reshape((N, N))
#         Y = self.FM.numpy().reshape((N, N))
#         zs = np.array(self.losses)
#         Z = zs.reshape(X.shape)
#         mesh_plot_3d(X, Y, Z, self.target_idx)

#     def contour_gradients(self):
#         X = self.AM.numpy().reshape((N, N))
#         Y = self.FM.numpy().reshape((N, N))
#         zs = np.array(self.losses)
#         Z = zs.reshape(X.shape)
#         plot_contour_gradient(
#             X,
#             Y,
#             Z,
#             self.target_idx,
#             self.grads
#         )