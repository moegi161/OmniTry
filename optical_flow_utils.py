import random
from typing import List, Optional

import torch
import torchvision
from torchvision.transforms import functional as TF


"""optical flow and trajectories sampling"""
def preprocess(img1_batch, img2_batch, video_size, transforms):
    img1_batch = torchvision.transforms.functional.resize(img1_batch, size=[video_size, video_size], antialias=False)
    img2_batch = torchvision.transforms.functional.resize(img2_batch, size=[video_size, video_size], antialias=False)
    return transforms(img1_batch, img2_batch)

def keys_with_same_value(dictionary):
    result = {}
    for key, value in dictionary.items():
        if value not in result:
            result[value] = [key]
        else:
            result[value].append(key)

    conflict_points = {}
    for k in result.keys():
        if len(result[k]) > 1:
            conflict_points[k] = result[k]
    return conflict_points

def find_duplicates(input_list):
    seen = set()
    duplicates = set()

    for item in input_list:
        if item in seen:
            duplicates.add(item)
        else:
            seen.add(item)

    return list(duplicates)

def neighbors_index(point, window_size, H, W):
    """return the spatial neighbor indices"""
    t, x, y = point
    neighbors = []
    for i in range(-window_size, window_size + 1):
        for j in range(-window_size, window_size + 1):
            if i == 0 and j == 0:
                continue
            if x + i < 0 or x + i >= H or y + j < 0 or y + j >= W:
                continue
            neighbors.append((t, x + i, y + j))
    return neighbors


@torch.no_grad()
def compute_raft_flows(
    frames: List["Image.Image"],
    device: torch.device,
    resize_to: Optional[int] = None,
):
    """
    Compute RAFT optical flow for a list of PIL frames.

    Returns a tensor of shape [T-1, 2, H, W] on CPU (float32).
    """
    from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

    if len(frames) < 2:
        return []

    # Convert to tensor batch [T, C, H, W]
    frame_tensors = torch.stack([TF.to_tensor(f) for f in frames])

    # Optional resize to keep flow inference cheap / match model input
    if resize_to is not None:
        frame_tensors = TF.resize(frame_tensors, size=[resize_to, resize_to], antialias=False)

    video_h = frame_tensors.shape[-2]
    weights = Raft_Large_Weights.DEFAULT
    transforms = weights.transforms()

    # Prepare consecutive pairs
    img1_batch, img2_batch = preprocess(frame_tensors[:-1], frame_tensors[1:], video_h, transforms)

    model = raft_large(weights=weights, progress=False).to(device)
    model = model.eval()

    flow_preds = model(img1_batch.to(device), img2_batch.to(device))
    flow = flow_preds[-1].cpu()  # (T-1, 2, H, W)
    return flow


@torch.no_grad()
def compute_flow_guidance(
    frames: List["Image.Image"],
    device: torch.device,
    resize_to: Optional[int] = None,
    max_neighbors: int = 2,
):
    """
    Build attention donors from RAFT flow magnitudes.

    Returns:
        dict with:
            - donors_temp: list[list[int]] length = T + 1 (targets + ref)
            - flow_fields: flow tensor (T-1, 2, H, W) or []
    """
    if len(frames) < 2:
        return None

    flows = compute_raft_flows(frames, device=device, resize_to=resize_to)
    if len(flows) == 0:
        return None

    # Mean motion per adjacent pair
    mean_mags = flows.abs().mean(dim=(1, 2, 3))
    num_targets = len(frames)
    ref_idx = num_targets  # reference sits at the end of the batch

    donors_temp = []
    for i in range(num_targets):
        candidates = []
        if i > 0:
            candidates.append((mean_mags[i - 1].item(), i - 1))
        if i < num_targets - 1:
            candidates.append((mean_mags[i].item(), i + 1))

        # pick the most stable neighbors (smallest motion)
        candidates = sorted(candidates, key=lambda x: x[0])[:max_neighbors]
        donors_temp.append([j for _, j in candidates] + [ref_idx])

    donors_temp.append([])  # ref has no donors

    return {"donors_temp": donors_temp, "flow_fields": flows}


@torch.no_grad()
def sample_trajectories(video_path, device, is_video=False):
    from torchvision.models.optical_flow import Raft_Large_Weights
    from torchvision.models.optical_flow import raft_large

    weights = Raft_Large_Weights.DEFAULT
    transforms = weights.transforms()

    if is_video:
        frames, _, _ = torchvision.io.read_video(str(video_path), output_format="TCHW")
    else:
        frames = video_path

    video_size = frames.shape[-1]
    print("video size", video_size)
    clips = list(range(len(frames)))

    model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device)
    model = model.eval()

    finished_trajectories = []

    current_frames, next_frames = preprocess(frames[clips[:-1]], frames[clips[1:]], video_size,transforms)
    list_of_flows = model(current_frames.to(device), next_frames.to(device))
    predicted_flows = list_of_flows[-1]

    predicted_flows = predicted_flows/video_size

    resolutions = [16,8] #[64, 32, 16, 8]
    res = {}
    window_sizes = {256:2,
                    128: 2,
                    64: 2,
                    32: 1,
                    16: 1,
                    8: 1}

    for resolution in resolutions:
        print("="*30)
        trajectories = {}
        predicted_flow_resolu = torch.round(resolution*torch.nn.functional.interpolate(predicted_flows, scale_factor=(resolution/video_size, resolution/video_size)))

        T = predicted_flow_resolu.shape[0]+1
        H = predicted_flow_resolu.shape[2]
        W = predicted_flow_resolu.shape[3]

        is_activated = torch.zeros([T, H, W], dtype=torch.bool)

        for t in range(T-1):
            flow = predicted_flow_resolu[t]
            for h in range(H):
                for w in range(W):

                    if not is_activated[t, h, w]:
                        is_activated[t, h, w] = True
                        # this point has not been traversed, start new trajectory
                        x = h + int(flow[1, h, w])
                        y = w + int(flow[0, h, w])
                        if x >= 0 and x < H and y >= 0 and y < W:
                            # trajectories.append([(t, h, w), (t+1, x, y)])
                            trajectories[(t, h, w)]= (t+1, x, y)

        conflict_points = keys_with_same_value(trajectories)
        for k in conflict_points:
            index_to_pop = random.randint(0, len(conflict_points[k]) - 1)
            conflict_points[k].pop(index_to_pop)
            for point in conflict_points[k]:
                if point[0] != T-1:
                    trajectories[point]= (-1, -1, -1) # stupid padding with (-1, -1, -1)

        active_traj = []
        all_traj = []
        for t in range(T):
            pixel_set = {(t, x//H, x%H):0 for x in range(H*W)}
            new_active_traj = []
            for traj in active_traj:
                if traj[-1] in trajectories:
                    v = trajectories[traj[-1]]
                    new_active_traj.append(traj + [v])
                    pixel_set[v] = 1
                else:
                    all_traj.append(traj)
            active_traj = new_active_traj
            active_traj+=[[pixel] for pixel in pixel_set if pixel_set[pixel] == 0]
        all_traj += active_traj

        useful_traj = [i for i in all_traj if len(i)>1]
        for idx in range(len(useful_traj)):
            if useful_traj[idx][-1] == (-1, -1, -1):
                useful_traj[idx] = useful_traj[idx][:-1]
        print("how many points in all trajectories for resolution{}?".format(resolution), sum([len(i) for i in useful_traj]))
        print("how many points in the video for resolution{}?".format(resolution), T*H*W)

        # validate if there are no duplicates in the trajectories
        trajs = []
        for traj in useful_traj:
            trajs = trajs + traj
        assert len(find_duplicates(trajs)) == 0, "There should not be duplicates in the useful trajectories."

        # check if non-appearing points + appearing points = all the points in the video
        all_points = set([(t, x, y) for t in range(T) for x in range(H) for y in range(W)])
        left_points = all_points- set(trajs)
        print("How many points not in the trajectories for resolution{}?".format(resolution), len(left_points))
        for p in list(left_points):
            useful_traj.append([p])
        print("how many points in all trajectories for resolution{} after pending?".format(resolution), sum([len(i) for i in useful_traj]))


        longest_length = max([len(i) for i in useful_traj])
        sequence_length = (window_sizes[resolution]*2+1)**2 + longest_length - 1

        seqs = []
        masks = []

        # create a dictionary to facilitate checking the trajectories to which each point belongs.
        point_to_traj = {}
        for traj in useful_traj:
            for p in traj:
                point_to_traj[p] = traj

        for t in range(T):
            for x in range(H):
                for y in range(W):
                    neighbours = neighbors_index((t,x,y), window_sizes[resolution], H, W)
                    sequence = [(t,x,y)]+neighbours + [(0,0,0) for i in range((window_sizes[resolution]*2+1)**2-1-len(neighbours))]
                    sequence_mask = torch.zeros(sequence_length, dtype=torch.bool)
                    sequence_mask[:len(neighbours)+1] = True

                    traj = point_to_traj[(t,x,y)].copy()
                    traj.remove((t,x,y))
                    sequence = sequence + traj + [(0,0,0) for k in range(longest_length-1-len(traj))]
                    sequence_mask[(window_sizes[resolution]*2+1)**2: (window_sizes[resolution]*2+1)**2 + len(traj)] = True

                    seqs.append(sequence)
                    masks.append(sequence_mask)

        seqs = torch.tensor(seqs)
        masks = torch.stack(masks)
        res["traj{}".format(resolution)] = seqs
        res["mask{}".format(resolution)] = masks
    return res
