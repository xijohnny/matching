from abc import abstractmethod
from typing import Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt
import torch
import zarr
from torchvision import transforms

class RandomRotateCrop(torch.nn.Module):
    def __init__(
        self,
        size: int,
        degrees: float = 90,
        interpolation: transforms.InterpolationMode = transforms.InterpolationMode.BILINEAR,
    ) -> None:
        super().__init__()
        self._FILL = -1
        self.random_rotate = transforms.RandomRotation(
            degrees=degrees,
            expand=True,
            interpolation=interpolation,
            fill=self._FILL,
        )
        self.random_crop = transforms.RandomCrop(size=size)

    def valid_crop(self, x: torch.Tensor) -> torch.Tensor:
        _x: torch.Tensor = self.random_crop(x)
        while _x.eq(self._FILL).any():
            _x = self.random_crop(x)
        return _x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _x = x.short()
        _x = self.random_rotate(_x)
        _x = self.valid_crop(_x)
        _x = _x.byte()
        return _x


class MinMaxScale(torch.nn.Module):
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # percentile scale
        # img = img.float()
        # _img = img.reshape(6, -1)
        # _min = _img.quantile(q=0.01, dim=1, keepdim=True).unsqueeze(dim=2)
        # _max = _img.quantile(q=0.99, dim=1, keepdim=True).unsqueeze(dim=2)

        # minmax scale
        _min = img.amin(dim=(1, 2), keepdim=True)
        _max = img.amax(dim=(1, 2), keepdim=True)
        img -= _min
        img = img.half()
        img /= _max - _min
        img *= 255.0
        return img.byte()


class ZarrCrop:
    def __init__(self, size: Union[int, Sequence[int], None] = None):
        """
        Base class for ZarrRandomCrop, ZarrCenterCrop, and ZarrLoadEntireArray

        Parameters
        ----------
        size : Union[int, Sequence[int], None]
            size for ZarrRandomCrop or ZarrCenterCrop. None for ZarrLoadEntireArray, by default None
        """

        #  whether ZarrCrop returns a tensor for single cell crops
        #     i.e. if False returns [C, H, W] if True [n_cells, C, H, W]
        self.single_cell_crops = False
        if size is not None:
            self.size = self._setup_size(size)

    @abstractmethod
    def get_params(self, img: zarr.core.Array, output_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
        pass

    @staticmethod
    def _setup_size(size: Union[int, Sequence[int]]) -> Tuple[int, int]:
        """
        converts size int or sequence to tuple([int,int])

        Parameters
        ----------
        size : int | Sequence[int]
            size to crop from Zarr
        error_msg : str
            if wrong format is given

        Returns
        -------
        Tuple[int, int]
            crop size in (H x W)

        Raises
        ------
        ValueError
            if wrong format is given
        """
        if isinstance(size, int):
            return size, size
        elif isinstance(size, Sequence) and len(size) == 1:
            return int(size[0]), int(size[0])
        elif isinstance(size, Sequence) and len(size) == 2:
            return int(size[0]), int(size[1])
        else:
            raise ValueError("Please provide only two dimensions (h, w) for size.")

    @staticmethod
    def _convert_to_tensor(input_array: np.ndarray) -> torch.Tensor:
        """
        converts numpy array to tensor and permutes channels from (H x W x C)->(C x H x W)

        Parameters
        ----------
        input_array : np.ndarray (H x W x C)

        Returns
        -------
        torch.Tensor (C x H x W)
        """
        tensor = torch.from_numpy(input_array)
        tensor = tensor.permute(2, 0, 1)
        return tensor.contiguous()  # make the permutation encoded into memory

    def __call__(self, zarr_array: zarr.core.Array) -> torch.Tensor:
        """
        Takes in a zarr array (H x W x C) and returns a random crop as a torch.Tensor (C x H x W)

        Parameters
        ----------
        zarr_array : zarr.core.Array
            input zarr array, typically from dataloader

        Returns
        -------
        torch.Tensor
            Center crop as a torch.Tensor (C x H x W)
        """
        top, left, height, width = self.get_params(zarr_array, self.size)  # get crop indices
        zarr_array_cropped = zarr_array[top : (top + height), left : (left + width), :]  # slice into zarr -> np.ndarray
        tensor = self._convert_to_tensor(zarr_array_cropped)  # convert numpy to tensor and swap channels
        return tensor


class ZarrRandomCrop(ZarrCrop):
    def __init__(self, size: Union[int, Sequence[int]]):
        """
        Re-implementation of torchvision.transforms.RandomCrop that can work with zarrs.
        This function can be wrapped with torchvision.transforms.Lambda and composed with
        torchvision.transforms.Compose but NOT with torch.nn.Sequential
        NOTE - Padding not supported in this version

        Parameters
        ----------
        size : Union[int, Tuple[int, int]]
            size of random crop
        """
        super().__init__(size)

    def get_params(self, img: zarr.core.Array, output_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """
        Get parameters for ``crop`` for a random crop.

        Parameters
        ----------
        img : zarr.core.Array
            Zarr array to be cropped
        output_size : Tuple[int, int]
            Expected output size of the crop

        Returns
        -------
        Tuple[int, int, int, int]
            params (top, left, height, width) to be used in ``self.__call__`` for generating crop.
        """
        height, width = output_size
        h, w, _ = img.shape

        if h < height or w < width:
            raise ValueError(f"Required crop size {(height, width)} is larger than input image size {(h, w)}")

        if w == width and h == height:
            return 0, 0, h, w

        top = np.random.randint(0, h - height, size=1)[0]
        left = np.random.randint(0, w - width, size=1)[0]
        return top, left, height, width


class ZarrCenterCrop(ZarrCrop):
    def __init__(self, size: Union[int, Sequence[int]]):
        """
        Re-implementation of torchvision.transforms.CenterCrop that can work with zarrs.
        This function can be wrapped with torchvision.transforms.Lambda and composed with
        torchvision.transforms.Compose but NOT with torch.nn.Sequential
        NOTE - Padding not supported in this version

        Parameters
        ----------
        size : Union[int, Tuple[int, int]]
            size of center crop
        """
        super().__init__(size)

    def get_params(self, img: zarr.core.Array, output_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """
        Get parameters for ``crop`` for a center crop.

        Parameters
        ----------
        img : zarr.core.Array
            Zarr array to be cropped
        output_size : Tuple[int, int]
            Expected output size of the crop

        Returns
        -------
        Tuple[int, int, int, int]
            params (top, left, height, width) to be used in ``self.__call__`` for generating crop.
        """
        height, width = output_size
        h, w, _ = img.shape

        if h < height or w < width:
            raise ValueError(f"Required crop size {(height, width)} is larger than input image size {(h, w)}")

        if w == width and h == height:
            return 0, 0, h, w

        top = int(round((h - height) / 2.0))
        left = int(round((w - width) / 2.0))

        return top, left, height, width


class ZarrLoadEntireArray(ZarrCrop):
    def __init__(self):
        """
        Loads entire zarr array into torch.Tensor and reformats to (C x H x W)
        """
        super().__init__()
        self.size = None  # type: ignore[assignment]

    def get_params(
        self, img: zarr.core.Array, output_size: None  # type: ignore[override]
    ) -> Tuple[int, int, int, int]:
        """
        Get parameters for ``crop`` for a entire zarr array.

        Parameters
        ----------
        img : zarr.core.Array
            Zarr array to be cropped
        output_size : None

        Returns
        -------
        Tuple[int, int, int, int]
            params (top, left, height, width) to be used in ``self.__call__`` for generating crop.
        """
        if output_size is not None:
            raise ValueError("Output size should not be defined for ZarrLoadEntireArray")

        height, width, _ = img.shape
        top = left = 0
        return top, left, height, width


### I think you need mahotas library for this

class ZarrSingleCellCrops:
    def __init__(
        self,
        zarr_crop: ZarrCrop = ZarrLoadEntireArray(),
        max_cells: int = 16,
        bbox_size: int = 128,
        jitter_center: int = 32,
        fill_missing_cells: bool = True,
        random_crop_rate: float = 0,
    ):
        """
        Class for extracing single cell crops

        Parameters
        ----------
        zarr_crop : ZarrCrop
            Initial ZarrCrop to take from zarr array (i.e. ZarrRandomCrop, ZarrCenterCrop, ZarrLoadEntireArray)
        max_cells: int
            Max number of cells to sample from each image
        bbox_size: int
            Fixed crop size around each cell
        jitter_center: int
            Random offset from nuclei center to use for x, y jitter
        fill_missing_cells: bool
            Fill returned tensor with random crops if num_cells < max_cells
        random_sample_rate
            Rate at which to sample random crops instead of nuclei centers
        """

        #  whether ZarrCrop returns a tensor for single cell crops
        #     i.e. if False returns [C, H, W] if True [n_cells, C, H, W]
        self.single_cell_crops = True
        self.zarr_crop = zarr_crop
        self.max_cells = max_cells
        self.bbox_size = bbox_size
        self.drop_edge_cells = True  # can't handle crops that extend beyond image boundaries
        self.jitter_center = jitter_center
        self.fill_missing_cells = fill_missing_cells
        self.random_crop_rate = random_crop_rate

    @staticmethod
    def _label_nuclei(input_array: npt.NDArray) -> npt.NDArray:
        """
        Label nuclei from image numpy array

        Parameters
        ----------
        input_array : npt.NDArray
            6 channel cropped image [C, H, W]

        Returns
        -------
        npt.NDArray [H, W]
            labeled nuclei image, where background is zero and each nuclei's pixels gets unique number
        """

        nuc_chan = input_array[0, :, :]

        # nuc mask
        # based on Mahotas library - https://mahotas.readthedocs.io/en/latest/labeled.html
        nuc_chan = mh.gaussian_filter(nuc_chan, 4)
        nuc_chan = nuc_chan > nuc_chan.mean()

        # watershed
        # based on https://mahotas.readthedocs.io/en/latest/distance.html
        distances = mh.stretch(mh.distance(nuc_chan))
        Bc = np.ones((9, 9))
        maxima = mh.morph.regmax(distances, Bc=Bc)
        spots, n_spots = mh.label(maxima, Bc=Bc)
        surface = distances.max() - distances
        areas = mh.cwatershed(surface, spots)
        areas *= nuc_chan

        areas = mh.labeled.filter_labeled(areas, remove_bordering=True)[0]
        return areas  # type: ignore[no-any-return]

    @staticmethod
    def _get_nuc_centers(labeled_nuclei: npt.NDArray) -> npt.NDArray:
        """
        Get center coordinates of labeled nuclei

        Parameters
        ----------
        labeled_nuclei : npt.NDArray [H, W]
            labeled nuclei image, where background is zero and each nuclei's pixels gets unique number

        Returns
        -------
        npt.NDArray [number_of_cells, 2 (x & y coordinates)]
            center coordinates for each nuclei
        """

        # get bouding box
        # https://mahotas.readthedocs.io/en/latest/api.html#mahotas.bbox
        bb = mh.labeled.bbox(labeled_nuclei)[1:]
        nuc_centers = np.vstack([np.int32((bb[:, 0] + bb[:, 1]) / 2), np.int32((bb[:, 2] + bb[:, 3]) / 2)]).T
        return nuc_centers

    def _drop_edge_cells(self, nuc_centers: npt.NDArray, h: int, w: int) -> npt.NDArray:
        """
        Drops nuclei coordinates who's bounding box extends beyond image

        Parameters
        ----------
        nuc_centers : npt.NDArray
            center coordinates for each nuclei
        h : int
            image height
        w : int
            image width
        Returns
        -------
        npt.NDArray
            center coordinates for each nuclei after filtering bounding boxes that extend beyond image
        """

        nuc2keep = (
            ((nuc_centers[:, 0] - self.bbox_size // 2 - self.jitter_center) > 0)
            & ((nuc_centers[:, 1] - self.bbox_size // 2 - self.jitter_center) > 0)
            & ((nuc_centers[:, 0] + self.bbox_size // 2 + self.jitter_center) < h)
            & ((nuc_centers[:, 1] + self.bbox_size // 2 + self.jitter_center) < w)
        )

        nuc_centers = nuc_centers[nuc2keep]

        return nuc_centers

    def _get_random_coordinates(self, h, w):
        """
        Get random x, y coordinate

        Parameters
        ----------
        h : int
            image height
        w : int
            image width
        Returns
        -------
        npt.NDArray
            Random coordinates for bounding boxes
        """

        return np.array(
            [
                np.random.randint(
                    self.bbox_size // 2 + self.jitter_center, h - self.bbox_size // 2 - self.jitter_center
                ),
                np.random.randint(
                    self.bbox_size // 2 + self.jitter_center, w - self.bbox_size // 2 - self.jitter_center
                ),
            ]
        )

    def _compute_cropped_array(self, cropped_array: npt.NDArray, nuc_center: npt.NDArray) -> npt.NDArray:
        """
        Returns single cell crop from image given cell coordinate

        Parameters
        ----------
        cropped_array : npt.NDArray
            input crop
        nuc_center : npt.NDArray
            nucleus center coordinate

        Returns
        -------
        npt.NDArray
            crop of size bounding_box around nuc_center coordinate - [C, H, W]
        """

        current_center = nuc_center + np.array(
            [
                self.jitter_center // 2 - np.random.randint(self.jitter_center),
                self.jitter_center // 2 - np.random.randint(self.jitter_center),
            ]
        )
        return cropped_array[
            :,
            (current_center[0] - self.bbox_size // 2) : (current_center[0] + self.bbox_size // 2),
            (current_center[1] - self.bbox_size // 2) : (current_center[1] + self.bbox_size // 2),
        ]

    def __call__(self, zarr_array: zarr.core.Array) -> torch.Tensor:
        """
        Get single cell crops from zarr_crop

        Parameters
        ----------
        zarr_array : zarr.core.Array
            loaded zarr array

        Returns
        -------
        torch.Tensor
            tensor with single cell crops - [number of cells, C, H, W]
        """

        cropped_tensor = self.zarr_crop(zarr_array)
        cropped_array = cropped_tensor.numpy()

        labeled_nuc = self._label_nuclei(cropped_array)
        nuc_centers = self._get_nuc_centers(labeled_nuc)

        c, h, w = cropped_array.shape

        if self.drop_edge_cells:
            nuc_centers = self._drop_edge_cells(nuc_centers, h, w)

        np.random.shuffle(nuc_centers)

        cropped_cells = np.zeros((self.max_cells, c, self.bbox_size, self.bbox_size), dtype=cropped_array.dtype)

        i = 0
        while (i < len(nuc_centers)) and (i < self.max_cells):
            if np.random.random_sample() < self.random_crop_rate:
                current_coords = self._get_random_coordinates(h, w)
            else:
                current_coords = nuc_centers[i]
            cropped_cells[i, :, :, :] = self._compute_cropped_array(cropped_array, current_coords)
            i += 1

        # if less than number of cells found fill with random crops
        if self.fill_missing_cells:
            while i < self.max_cells:
                random_center = self._get_random_coordinates(h, w)
                cropped_cells[i, :, :, :] = self._compute_cropped_array(cropped_array, random_center)
                i += 1

        cropped_cells = cropped_cells[:i]

        cropped_cells = torch.from_numpy(cropped_cells)

        return cropped_cells.contiguous()
