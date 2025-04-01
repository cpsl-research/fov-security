import torch
from collections import defaultdict
from tqdm import tqdm

from torchvision import transforms

from fov.train import BinarySegmentation


def get_metrics(pc_img_pred, gt_mask, metadata, threshold: float = 0.7):
    results = defaultdict()

    # metrics
    metrics = BinarySegmentation.metrics(
        outputs=pc_img_pred,
        labels=gt_mask,
        loss_fn=None,
        threshold=threshold,
        pos_weight=1.0,
        neg_weight=1.0,
        run_auprc=True,
    )

    # add the results to the logger
    results["adv_model"] = metadata["adv_model"]
    results["n_pts_adv"] = metadata["n_pts_adv"]
    results["frame"] = metadata["frame"]
    results["sensor"] = metadata["sensor"]
    results["agent"] = metadata["agent"]

    # # add pretty results specifically for plotting
    # results["Adversary Model"].append(adv_titles[metadata["adv_model"]])
    # results["Model Name"].append(model_titles[model_name])

    # add quantitative metrics
    for k, v in metrics.items():
        if v is not None:
            try:
                try:
                    res = v.detach().cpu().item()
                except ValueError:
                    res = v.detach().cpu()
                results[k] = res
            except AttributeError:
                results[k] = v

    return results


def test_model_on_dataset(
    model,
    name_model_dataset: str,
    seg_dataset,
    name_test_dataset: str,
    n_frames_max: int,
):
    """Test the model on the provided dataset"""
        
    # set the results data structure
    results = defaultdict(list)  # set the results for this combo
    
    # run the evaluation
    print(
        f"\n----Evaluating model from {name_model_dataset.upper()} "
        f"on {name_test_dataset.upper()} data"
    )
    n_frames_run = min(n_frames_max, len(seg_dataset))
    i_index = 0
    i_run = 0
    with tqdm(total=n_frames_run, desc="Processing") as pbar:
        while i_run < n_frames_run:
            # only test adversarial on adversarial
            if i_index < len(seg_dataset):
                metadata = seg_dataset.get_metadata(i_index)
                if "adv" in name_test_dataset.lower():
                    if metadata["adv_model"] != "uniform":
                        i_index += 1
                        continue

                # get data
                pc_img, gt_mask = seg_dataset[i_index]
                pc_img = torch.unsqueeze(pc_img, 0)
                pc_np = seg_dataset.get_pointcloud(i_index)

                # inference
                pc_img_pred = model(pc_img, pc_np, metadata)

                # metrics
                this_results = get_metrics(pc_img_pred, gt_mask, metadata)
                for k, v in this_results.items():
                    results[k].append(v)

                # check if done
                pbar.update(1)
                i_run += 1
                i_index += 1
            else:
                print("Index exceeds dataset length...stopping")
                break
    return results


def test_model_on_dataset_v2(
    model,
    name_model_dataset: str,
    seg_dataset,
    name_test_dataset: str,
    n_frames_max: int,
    evaluate_at_resolution: bool = False,
    dataset_at_resolution = None,
):
    """Test the model on the provided dataset"""
        
    # set the results data structure
    results = []  # set the results for this combo
    results_res = []

    # run the evaluation
    print(
        f"\n----Evaluating model from {name_model_dataset.upper()} "
        f"on {name_test_dataset.upper()} data"
    )
    n_frames_run = min(n_frames_max, len(seg_dataset))
    i_index = 0
    i_run = 0
    with tqdm(total=n_frames_run, desc="Processing") as pbar:
        while i_run < n_frames_run:
            # only test adversarial on adversarial
            if i_index < len(seg_dataset):
                metadata = seg_dataset.get_metadata(i_index)
                if "adv" in name_test_dataset.lower():
                    if metadata["adv_model"] != "uniform":
                        i_index += 1
                        continue

                # get data
                pc_img, gt_mask = seg_dataset[i_index]
                pc_img = torch.unsqueeze(pc_img, 0)
                pc_np = seg_dataset.get_pointcloud(i_index)

                # inference
                pc_img_pred = model(pc_img, pc_np, metadata)

                # metrics
                this_results = get_metrics(pc_img_pred, gt_mask, metadata)
                meta_augment = ["frame", "agent", "sensor", "attacked"]
                for meta in meta_augment:
                    this_results[meta] = metadata[meta]
                results.append(this_results)

                # evaluate at a particular resolution
                if evaluate_at_resolution:
                    # get the gt mask at this resolution
                    _, gt_mask_res = dataset_at_resolution[i_index]

                    # scale the inference to get to this resolution
                    pc_img_pred_res = transforms.Resize(
                        size=gt_mask_res.shape[1:]
                    )(pc_img_pred)

                    # get the results at this resolution
                    this_results_res = get_metrics(pc_img_pred_res, gt_mask_res, metadata)
                    results_res.append(this_results_res)

                # check if done
                pbar.update(1)
                i_run += 1
                i_index += 1
            else:
                print("Index exceeds dataset length...stopping")
                break

    return results, results_res