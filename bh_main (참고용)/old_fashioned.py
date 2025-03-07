
import random

from jinja2 import pass_environment

from modules.data_loader import *
from modules.network import *
from modules.neuron import *
from modules.synapse import *
from modules.old_fashioned import *
from modules.ae_network import *

from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from scipy.optimize import linear_sum_assignment
import itertools


def seed_assign(seed):
    random.seed(seed)                          # Python random 시드 고정
    np.random.seed(seed)                       # NumPy 시드 고정
    torch.manual_seed(seed)                    # PyTorch CPU 시드 고정
    torch.cuda.manual_seed(seed)               # PyTorch GPU 시드 고정
    torch.cuda.manual_seed_all(seed)           # PyTorch 멀티 GPU 시드 고정
    torch.backends.cudnn.deterministic = True  # 연산의 결정론적 동작 보장
    # torch.backends.cudnn.benchmark = False     # 성능 최적화 비활성화 (결정론적 보장)


def plot_distributions(ds, plot_tau, plot_denominator, plot_m, plot_max_tau, cos_thr_ds,
                       tr_cycle_acc, post_tr_cycle_acc, total_cycle_acc):
    """
    plot_tau, plot_denominator, plot_m, plot_max_tau 값을 시각화하는 함수.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # 1. Tau 값의 히스토그램
    plt.figure(figsize=(10, 5))
    plt.hist(plot_tau, bins=50, alpha=0.7, color='blue', label='tau')
    plt.axvline(np.mean(plot_tau), color='red', linestyle='dashed', linewidth=1, label='Mean Tau')
    plt.xlabel('Tau Values')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Tau Values, dataset: {ds}\n'
              f'tr_cycle_acc: {tr_cycle_acc:.2f}, post_tr_cycle_acc: {post_tr_cycle_acc:.2f}, '
              f'total_cycle_acc: {total_cycle_acc:.2f}')
    plt.legend()
    plt.show()

    # 2. Denominator 값의 히스토그램
    plt.figure(figsize=(10, 5))
    plt.hist(plot_denominator, bins=50, alpha=0.7, color='orange', label='denominator')
    plt.axvline(np.mean(plot_denominator), color='red', linestyle='dashed', linewidth=1, label='Mean Denominator')
    plt.xlabel('Denominator Values')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Denominator Values, dataset: {ds}\n'
              f'tr_cycle_acc: {tr_cycle_acc:.2f}, post_tr_cycle_acc: {post_tr_cycle_acc:.2f}, '
              f'total_cycle_acc: {total_cycle_acc:.2f}')
    plt.legend()
    plt.show()

    # 3. M 값의 히스토그램 (클러스터 인덱스 분포)
    plt.figure(figsize=(10, 5))
    unique_m, counts_m = np.unique(plot_m, return_counts=True)
    plt.bar(unique_m, counts_m, alpha=0.7, color='purple')
    plt.xlabel('Cluster Index (M)')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Cluster Indices (M), dataset: {ds}\n'
              f'tr_cycle_acc: {tr_cycle_acc:.2f}, post_tr_cycle_acc: {post_tr_cycle_acc:.2f}, '
              f'total_cycle_acc: {total_cycle_acc:.2f}')
    plt.xticks(unique_m)  # 클러스터 인덱스 표시
    plt.grid(axis='y')
    plt.show()

    # 4. Max Tau 값의 히스토그램
    cos_thr_percentile = 100 - (np.sum(np.array(plot_max_tau) < cos_thr_ds) / len(plot_max_tau) * 100)
    plt.figure(figsize=(10, 5))
    plt.hist(plot_max_tau, bins=50, alpha=0.7, color='green', label='max_tau')
    plt.axvline(cos_thr_ds, color='red', linestyle='dashed', linewidth=1, label=f'cos_thr: {cos_thr_ds:.2f}')
    plt.xlabel('Max Tau Values')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Max Tau Values, dataset: {ds}\n'
              f'cos_thr_ds is in the top {cos_thr_percentile:.2f}% of max_tau\n'
              f'tr_cycle_acc: {tr_cycle_acc:.2f}, post_tr_cycle_acc: {post_tr_cycle_acc:.2f}, '
              f'total_cycle_acc: {total_cycle_acc:.2f}')
    plt.legend()
    plt.text(
        cos_thr_ds, plt.gca().get_ylim()[1] * 0.9,
        f'{cos_thr_percentile:.2f}% (cos_thr_ds: {cos_thr_ds:.2f})',
        color='red', fontsize=12, verticalalignment='top'
    )
    plt.show()

    # 5. Max Tau vs. Cosine Threshold 비교 (산점도)
    plt.figure(figsize=(10, 5))
    plt.plot(plot_max_tau, marker='o', linestyle='', color='blue', alpha=0.5, label='Max Tau')
    plt.axhline(cos_thr_ds, color='red', linestyle='dashed', linewidth=1, label=f'cos_thr: {cos_thr_ds:.2f}')
    plt.xlabel('Spike Index')
    plt.ylabel('Max Tau Value')
    plt.title(f'Max Tau Values vs. Cosine Threshold, dataset: {ds}\n'
              f'cos_thr_ds is in the top {cos_thr_percentile:.2f}% of max_tau\n'
              f'tr_cycle_acc: {tr_cycle_acc:.2f}, post_tr_cycle_acc: {post_tr_cycle_acc:.2f}, '
              f'total_cycle_acc: {total_cycle_acc:.2f}')
    plt.text(
        len(plot_max_tau) * 0.7, cos_thr_ds, f'{cos_thr_percentile:.2f}%', color='red', fontsize=12, verticalalignment='bottom'
    )
    plt.legend()
    plt.grid()
    plt.show()

def plot_spike(spike, title="Spike Visualization (Black & White)"):
    """
    Spike 데이터를 검은색으로 시각화하는 함수.
    가로축: Timestep
    세로축: Feature
    각 요소를 구분하는 격자가 추가되며, x축과 y축에 일정한 간격으로 숫자 눈금을 표시함.
    """
    spike = spike.squeeze()
    spike[:, :] = spike[:, ::-1]  # Flip horizontally

    plt.figure(figsize=(10, 6))
    plt.imshow(spike.T, aspect='auto', cmap='Greys', interpolation='nearest')

    # x축 눈금 설정 (Timestep)
    x_ticks = np.arange(spike.shape[0])
    plt.xticks(x_ticks)  # X축 눈금 설정

    # y축 눈금을 feature 개수만큼 설정
    y_ticks = np.arange(spike.shape[1])
    plt.yticks(y_ticks)  # Y축 눈금 설정

    # 격자 추가
    plt.grid(visible=True, which='minor', color='gray', linestyle='--', linewidth=0.5)
    plt.gca().set_xticks(np.arange(-0.5, spike.shape[0], 1), minor=True)
    plt.gca().set_yticks(np.arange(-0.5, spike.shape[1], 1), minor=True)
    plt.grid(which='minor', color='gray', linestyle='--', linewidth=0.5)

    # 색상 막대와 라벨 추가
    plt.colorbar(label='Spike Value')
    plt.xlabel('Timestep')
    plt.ylabel('Feature')
    plt.title(title)

    plt.show()


def plot_origin_spike(spike, min_max_y_on = False):
    # 최소값과 최대값 계산
    if min_max_y_on == True:
        min_val = np.min(spike)
        max_val = np.max(spike)
    else:
        min_val = -2
        max_val = 2
    # min_val = 0
    # max_val = 1

    # 플로팅
    plt.figure(figsize=(8, 4))
    plt.plot(spike, marker='o', linestyle='-', color='b', label='Spike Feature')
    plt.title('Original Spike')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.ylim(min_val, max_val)  # y축 범위를 min_val에서 max_val로 설정
    plt.legend()
    plt.grid(True)
    plt.show()



def cluster_spikes_with_accuracy_torch(features, true_labels, n_clusters, init_point=None, reset_num = 10):
    """
    Perform k-means clustering and calculate clustering accuracy using PyTorch.

    Parameters:
        features (torch.Tensor): 2D tensor with shape [spike_num, feature].
        true_labels (torch.Tensor): 1D tensor with the true labels for each spike.
        n_clusters (int): The number of clusters for k-means.
        init_point (torch.Tensor or None): Initial centroids for k-means, if provided.

    Returns:
        tuple: (cluster_labels, accuracy)
            - cluster_labels (torch.Tensor): Cluster labels for each spike.
            - accuracy (float): Clustering accuracy.
    """
    true_labels = torch.from_numpy(true_labels).to(features.device)

    def kmeans(features, n_clusters, init_point, max_iter=1000, tol=1e-4):
        # Initialize centroids
        if init_point is None:
            centroids = features[torch.randperm(features.size(0))[:n_clusters]]
        else:
            centroids = init_point
        
        for _ in range(max_iter):
            # Compute distances and assign clusters
            distances = torch.cdist(features, centroids)
            labels = torch.argmin(distances, dim=1)
            
            # Update centroids
            new_centroids = torch.stack([features[labels == i].mean(dim=0) for i in range(n_clusters)])
            
            # Check for convergence
            if torch.allclose(centroids, new_centroids, atol=tol):
                break
            centroids = new_centroids

        return labels, centroids

    max_acc_best = 0
    for i in range(reset_num):
        # Perform k-means clustering
        cluster_labels, _ = kmeans(features, n_clusters, init_point)
        
        # Convert cluster labels for accuracy calculation
        cluster_labels_one_start = cluster_labels + 1  # [0, 1, 2] -> [1, 2, 3]
        label_converter_ground = list(range(1, n_clusters + 1))  # [1, 2, 3]
        label_converter_permutations = list(itertools.permutations(label_converter_ground))  # All permutations
        
        acc_bin = []
        for perm in label_converter_permutations:
            label_converter = torch.tensor(perm, dtype=torch.int32, device=features.device)
            acc = torch.sum(label_converter[cluster_labels_one_start - 1] == (true_labels +1)).item()
            acc_bin.append(acc)
        
        max_acc = max(acc_bin) / len(true_labels)
        if max_acc_best < max_acc:
            max_acc_best = max_acc
            cluster_labels_best = cluster_labels

    return max_acc_best

def cluster_spikes_with_accuracy(features, true_labels, n_clusters, init_point):
    """
    Perform k-means clustering and calculate clustering accuracy.

    Parameters:
        features (numpy.ndarray): 2D array with shape [spike_num, feature].
        true_labels (numpy.ndarray): 1D array with the true labels for each spike.
        n_clusters (int): The number of clusters for k-means.

    Returns:
        tuple: (cluster_labels, accuracy)
            - cluster_labels (numpy.ndarray): Cluster labels for each spike.
            - accuracy (float): Clustering accuracy.
    """
    # Perform k-means clustering
    if init_point is None:
        kmeans = KMeans(n_clusters=n_clusters,n_init='auto', random_state=42)
    else:
        init_point = init_point[:n_clusters]
        kmeans = KMeans(n_clusters=n_clusters, init=init_point, n_init=1, random_state=42)
    cluster_labels = kmeans.fit_predict(features)

    cluster_labels_one_start = cluster_labels + 1 # [0, 1, 2] -> [1, 2, 3]로 변환
    label_converter_ground = list(range(1, n_clusters + 1)) # [1, 2, 3] 생성
    label_converter_permutations = list(itertools.permutations(label_converter_ground)) # 모든 순열 구하기
    acc_bin = []
    for perm in label_converter_permutations:
        label_converter = list(perm)
        acc = 0    
        for i in range(len(features)):
            if(label_converter[int(cluster_labels_one_start[i]-1)] == true_labels[i]+1):
                acc += 1
        acc_bin.append(acc)
    max_acc = max(acc_bin)/len(true_labels)

    return max_acc

# Example usage:
# features = np.random.rand(100, 4)  # Replace with your spike feature array
# true_labels = np.random.randint(0, 3, 100)  # Replace with your true labels
# cluster_labels, accuracy = cluster_spikes_with_accuracy(features, true_labels, n_clusters=3)
# print("Cluster Labels:", cluster_labels)
# print("Accuracy:", accuracy)


def zero_to_one_normalize_features(spike, level_num, coarse_com_config, scaling=1.0, norm01=True):
    if norm01 == False:
        # print('level_num', level_num)
        spike = spike * scaling
        levels = torch.linspace(coarse_com_config[1], coarse_com_config[0], level_num, device=spike.device)  # 0에서 1까지 균등한 level_num 개의 값
        # print('levels', levels, len(levels))
        bucketized = torch.bucketize(spike, levels)
        # print('bucketized', bucketized[0])
        bucketized[bucketized == 0] = 1 # bucketize 결과에서 0인 요소를 1로 변경
        # print('bucketized2', bucketized[0])
        spike_normalized = levels[bucketized - 1] # 최종 양자화된 값 매핑
        # print('spike_normalized[0]', spike_normalized[0])
        # print('spike_ori', spike[0])
        # plot_origin_spike(spike_normalized[0].cpu().detach().numpy(), min_max_y_on=False)
    else:

        min_val = torch.min(spike, dim=1, keepdim=True)[0]
        max_val = torch.max(spike, dim=1, keepdim=True)[0]
        
        # Min-Max Normalization
        spike_normalized = (spike - min_val) / (max_val - min_val + 1e-12)

        spike_normalized = spike_normalized * scaling

        ## 실험용 ### ### ### ###
        # -0.5부터 0.5로 정규화
        # spike_normalized = spike_normalized
        ## 실험용 ###############

        # Quantization: level_num 단계로 매핑
        if level_num > 0:
            levels = torch.linspace(0, 1, level_num, device=spike.device)  # 0에서 1까지 균등한 level_num 개의 값
            bucketized = torch.bucketize(spike_normalized, levels)
            bucketized[bucketized == 0] = 1 # bucketize 결과에서 0인 요소를 1로 변경
            spike_normalized = levels[bucketized - 1] # 최종 양자화된 값 매핑
    # plot_origin_spike(spike_normalized[0].cpu().detach().numpy(), min_max_y_on=True)
    return spike_normalized



def copy_weights(source_encoder, target_encoder):
    """
    Copy weights and bias from source encoder to target encoder.
    Matches layers based on compatible weight shapes.
    """
    source_layers = list(source_encoder.named_modules())
    target_layers = list(target_encoder.named_modules())

    # Iterate through source and target layers
    src_i = 0
    tgt_i = 0
    # for src_i in range(len(source_layers)):
        # for tgt_i in range(len(target_layers)):
    
    while(src_i < len(source_layers) and tgt_i < len(target_layers)):
        src_name, src_layer = source_layers[src_i]
        tgt_name, tgt_layer = target_layers[tgt_i]

        if isinstance(src_layer, (nn.Conv1d, nn.Linear, nn.ConvTranspose1d)):
            if isinstance(tgt_layer, (nn.Conv1d, nn.Linear, nn.ConvTranspose1d)):
                src_i += 1
                tgt_i += 1
                # print(f"Copy layer: {src_name}, {src_layer} -> {tgt_name}, {tgt_layer}")
            else:
                tgt_i += 1
                continue
        else:
            src_i += 1
            continue
        
        # Check if weight shapes match
        if src_layer.weight.shape == tgt_layer.weight.shape:
            tgt_layer.weight.data = src_layer.weight.data.clone()
            # print(f"Copied weights: {src_name}, {src_layer} -> {tgt_name}, {tgt_layer}")
        else:
            assert False, f"Weight shape mismatch: {src_name}, {src_layer} -> {tgt_name}, {tgt_layer}"
        
        # Copy bias if it exists in both layers
        if (
            src_layer.bias is not None and 
            tgt_layer.bias is not None and 
            src_layer.bias.shape == tgt_layer.bias.shape
        ):
            tgt_layer.bias.data = src_layer.bias.data.clone()
            # print(f"Copied bias: {src_name}, {src_layer} -> {tgt_name}, {tgt_layer}")


def map_and_load_weights(saved_state_dict, current_state_dict):
    """
    Maps weights from a saved state_dict to a target network's state_dict.
    
    Args:
        saved_state_dict (dict): State dictionary from the saved model.
        current_state_dict (dict): State dictionary of the current network.

    Returns:
        dict: Updated state_dict for the current network.
    """
    # Get keys from both state_dicts
    saved_keys = list(saved_state_dict.keys())
    current_keys = list(current_state_dict.keys())
    
    # Initialize layer indices
    src_i = 0
    tgt_i = 0

    # Iterate over saved and target state_dict keys to map weights
    while src_i < len(saved_keys) and tgt_i < len(current_keys):
        src_layer = saved_keys[src_i]
        tgt_layer = current_keys[tgt_i]
        if 'weight' in src_layer or 'bias' in src_layer:
            if 'weight' in tgt_layer or 'bias' in tgt_layer:
                # Map weights and biases
                print(f"Copy layer: {src_layer} {saved_state_dict[src_layer].shape} -> {tgt_layer} {current_state_dict[tgt_layer].shape}")
                current_state_dict[tgt_layer] = saved_state_dict[src_layer]
                src_i += 1
                tgt_i += 1
            else:
                # Skip unmatched target layers
                tgt_i += 1
        else:
            # Skip unmatched source layers
            src_i += 1

    return current_state_dict


# 1 epoch training하고 activation 읽어라.
def plot_activation_distribution(model):
    total_activation_values_only_encoder = []
    total_activation_values = []
    for idx, layer in enumerate(model.module.encoder):
        if str(layer) == 'SSBH_activation_collector()':
            print(str(layer))
            activations = np.concatenate([in_layer.cpu().detach().numpy() for in_layer in layer.activation], axis=0)
            activation_values = activations.flatten()
            total_activation_values.append(activation_values)
            total_activation_values_only_encoder.append(activation_values)
            # 최대값, 평균, 분산 계산
            max_val = activation_values.max()
            mean_val = activation_values.mean()
            var_val = activation_values.var()
            
            # 히스토그램 그리기
            plt.figure(figsize=(10, 6))
            plt.hist(activation_values, bins=100, alpha=0.7, label=f'Layer {idx}')
            plt.title(f'Layer {str(layer)} - Max: {max_val:.2f}, Mean: {mean_val:.2f}, Variance: {var_val:.2f}')
            plt.xlabel('Activation Value')
            plt.ylabel('Frequency')
            plt.legend()
            plt.show()
        else:
            print('skip',str(layer))
    
    for idx, layer in enumerate(model.module.decoder):
        if str(layer) == 'SSBH_activation_collector()':
            
            print(str(layer))
            activations = np.concatenate([in_layer.cpu().detach().numpy() for in_layer in layer.activation], axis=0)
            activation_values = activations.flatten()
            total_activation_values.append(activation_values)
            
            # 최대값, 평균, 분산 계산
            max_val = activation_values.max()
            mean_val = activation_values.mean()
            var_val = activation_values.var()
            
            # 히스토그램 그리기
            plt.figure(figsize=(10, 6))
            plt.hist(activation_values, bins=100, alpha=0.7, label=f'Layer {idx}')
            plt.title(f'Layer {str(layer)} - Max: {max_val:.2f}, Mean: {mean_val:.2f}, Variance: {var_val:.2f}')
            plt.xlabel('Activation Value')
            plt.ylabel('Frequency')
            plt.legend()
            plt.show()




    # Plot for total activation values in encoder layers
    total_activation_values_only_encoder = np.concatenate(total_activation_values_only_encoder, axis=0)
    percentile_list = [95, 99, 99.7, 99.9, 99.99]
    # Calculate percentiles
    percentiles = [np.percentile(total_activation_values_only_encoder, p) for p in percentile_list]

    # Plotting the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(total_activation_values_only_encoder, bins=100, alpha=0.7, label='Total Encoder Activations')

    # Construct the title with percentile values
    percentile_text = ', '.join([f'{p}%: {percentiles[idx]:.2f}' for idx, p in enumerate(percentile_list)])
    plt.title(f'Total Encoder Activations - Max: {total_activation_values_only_encoder.max():.2f}, '
            f'Mean: {total_activation_values_only_encoder.mean():.2f}, '
            f'Variance: {total_activation_values_only_encoder.var():.2f}, '
            f'\nPercentile {percentile_text}')
    plt.xlabel('Activation Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()


    # # Plot for total activation values (both encoder and decoder)
    # total_activation_values = np.concatenate(total_activation_values, axis=0)
    # plt.figure(figsize=(10, 6))
    # plt.hist(total_activation_values, bins=100, alpha=0.7, label='Total Activations (Encoder + Decoder)')
    # plt.title(f'Total Activations - Max: {total_activation_values.max():.2f}, '
    #           f'Mean: {total_activation_values.mean():.2f}, '
    #           f'Variance: {total_activation_values.var():.2f}')
    # plt.xlabel('Activation Value')
    # plt.ylabel('Frequency')
    # plt.legend()
    # plt.show()


def l2_norm_loss(encoded_spike, target_norm=1.0):
    """
    각 feature 벡터의 L2 norm을 target_norm으로 맞추기 위한 Loss를 계산합니다.
    :param encoded_spike: [batch, feature] 차원의 텐서
    :param target_norm: 목표 L2 norm 크기 (default: 1.0)
    :return: L2Norm Loss
    """
    norms = encoded_spike.norm(p=2, dim=1)  # 각 feature 벡터의 L2 norm 계산
    loss = ((norms - target_norm) ** 2).mean()  # 목표 norm과의 차이 제곱합
    return loss

















########### dvs 데이터 시각화 코드#####################################################
########### dvs 데이터 시각화 코드#####################################################
########### dvs 데이터 시각화 코드#####################################################
# mapping = {0: 'Hand Clapping',1: 'Right Hand Wave',2: 'Left Hand Wave',3: 'Right Arm CW',4: 'Right Arm CCW',5: 'Left Arm CW',6: 'Left Arm CCW',7: 'Arm Roll',8: 'Air Drums',9: 'Air Guitar',10: 'Other'}
def dvs_visualization(inputs, labels, TIME, BATCH, my_seed):
    if inputs.size(4)==128:
        print('\n\n 128x128 data 32x32로 maxpool 해서 보여줄게 \n\n')
        timestep, batch_size, in_c, h, w = inputs.shape
        maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        inputs = maxpool(inputs.reshape(timestep*batch_size, in_c, h, w))
        inputs = maxpool(inputs).reshape(timestep, batch_size, in_c, h//4, w//4)
    seed_assign(seed = my_seed)
    what_input = random.randint(0, BATCH - 1)
    inputs_for_view = inputs.permute(1, 0, 2, 3, 4)
    for i in range(TIME):
        # 예시 데이터 생성
        data1 = inputs_for_view[what_input][i][0].numpy()  # torch tensor를 numpy 배열로 변환
        data2 = inputs_for_view[what_input][i][1].numpy()  # torch tensor를 numpy 배열로 변환

        # 데이터 플로팅
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # 1행 2열의 subplot 생성

        # 첫 번째 subplot에 데이터1 플로팅
        im1 = axs[0].imshow(data1, cmap='viridis', interpolation='nearest')
        axs[0].set_title(f'Channel 0\nLabel: {labels[what_input]}  Time: {i}')  # 라벨값 맵핑하여 제목에 추가
        axs[0].set_xlabel('X axis')
        axs[0].set_ylabel('Y axis')
        axs[0].grid(False)
        fig.colorbar(im1, ax=axs[0])  # Color bar 추가

        # 두 번째 subplot에 데이터2 플로팅
        im2 = axs[1].imshow(data2, cmap='viridis', interpolation='nearest')
        axs[1].set_title(f'Channel 1\nLabel: {labels[what_input]}  Time: {i}')  # 라벨값 맵핑하여 제목에 추가
        axs[1].set_xlabel('X axis')
        axs[1].set_ylabel('Y axis')
        axs[1].grid(False)
        fig.colorbar(im2, ax=axs[1])  # Color bar 추가

        plt.tight_layout()  # subplot 간 간격 조정
        plt.show()
    sys.exit("정상 종료")

########### dvs 데이터 시각화 코드#####################################################
########### dvs 데이터 시각화 코드#####################################################
########### dvs 데이터 시각화 코드#####################################################


##### First OTTT Synapse ###########################################################
##### First OTTT Synapse ###########################################################
##### First OTTT Synapse ###########################################################      
class SYNAPSE_CONV(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, trace_const1=1, trace_const2=0.7, TIME=8, OTTT_sWS_on = False, first_conv = False):
        super(SYNAPSE_CONV, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.trace_const1 = trace_const1
        self.trace_const2 = trace_const2
        # self.weight = torch.randn(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size, requires_grad=True)
        # self.bias = torch.randn(self.out_channels, requires_grad=True)
        self.weight = nn.Parameter(torch.randn(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
        self.bias = nn.Parameter(torch.randn(self.out_channels))
        # Kaiming 초기화
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.bias, 0)

        self.TIME = TIME

        self.OTTT_sWS_on = OTTT_sWS_on
        self.first_conv = first_conv 

        if (self.OTTT_sWS_on == True):
            self.gain = nn.Parameter(torch.ones(self.out_channels, 1, 1, 1))

    def forward(self, spike):
        # spike: [Time, Batch, Channel, Height, Width]   
        # print('spike.shape', spike.shape)
        Time = spike.shape[0]
        assert Time == self.TIME, f'Time dimension {Time} should be same as TIME {self.TIME}'
        Batch = spike.shape[1] 
        Channel = self.out_channels
        Height = (spike.shape[3] + self.padding*2 - self.kernel_size) // self.stride + 1
        Width = (spike.shape[4] + self.padding*2 - self.kernel_size) // self.stride + 1

        WS_weight = self.weight
        if (self.OTTT_sWS_on == True):
            fan_in = np.prod(self.weight.shape[1:])
            mean = torch.mean(self.weight, axis=[1, 2, 3], keepdims=True)
            var = torch.var(self.weight, axis=[1, 2, 3], keepdims=True)
            WS_weight = (self.weight - mean) / ((var * fan_in + 1e-4) ** 0.5)
            WS_weight = WS_weight * self.gain



        # output_current = torch.zeros(Time, Batch, Channel, Height, Width, device=spike.device)
        output_current = []
        
        # spike_detach = spike.detach().clone()
        spike_detach = spike.detach()
        spike_past = torch.zeros_like(spike_detach[0],requires_grad=False)
        spike_now = torch.zeros_like(spike_detach[0],requires_grad=False)
        for t in range(Time):
            # print(f'time:{t}', torch.sum(spike_detach[t]/ torch.numel(spike_detach[t])))
            spike_now = self.trace_const1*spike_detach[t] + self.trace_const2*spike_past

            # output_current[t]= SYNAPSE_CONV_METHOD.apply(spike[t], spike_now, self.weight, self.bias, self.stride, self.padding) 
            
            if (self.first_conv == True):  
                print('first_conv 확인했냐???\n')
                spike_now = spike[t].detach()
            output_current.append( SYNAPSE_CONV_METHOD.apply(spike[t], spike_now, WS_weight, self.bias, self.stride, self.padding) )
            
            spike_past = spike_now
            # print(f'time:{t}', torch.sum(output_current[t]/ torch.numel(output_current[t])))

        output_current = torch.stack(output_current, dim=0)
        return output_current

class SYNAPSE_CONV_METHOD(torch.autograd.Function):
    @staticmethod
    def forward(ctx, spike_one_time, spike_now, weight, bias, stride=1, padding=1):
        ctx.save_for_backward(spike_one_time, spike_now, weight, bias, torch.tensor([stride], requires_grad=False), torch.tensor([padding], requires_grad=False))
        return F.conv2d(spike_one_time, weight, bias=bias, stride=stride, padding=padding)

    @staticmethod
    def backward(ctx, grad_output_current):
        spike_one_time, spike_now, weight, bias, stride, padding = ctx.saved_tensors
        stride=stride.item()
        padding=padding.item()
        
        ## 이거 클론해야되는지 모르겠음!!!!
        grad_output_current_clone = grad_output_current.clone()


        grad_input_spike = grad_weight = grad_bias = None


        if ctx.needs_input_grad[0]:
            grad_input_spike = F.conv_transpose2d(grad_output_current_clone, weight, stride=stride, padding=padding)
        if ctx.needs_input_grad[2]:
            grad_weight = torch.nn.grad.conv2d_weight(spike_now, weight.shape, grad_output_current_clone,
                                                    stride=stride, padding=padding)
        if bias is not None and ctx.needs_input_grad[3]:
            grad_bias = grad_output_current_clone.sum((0, -1, -2))


        # print('grad_input_spike_conv', grad_input_spike)
        # print('grad_weight_conv', grad_weight)
        # print('grad_bias_conv', grad_bias)
        # print('grad_input_spike_conv', ctx.needs_input_grad[0])
        # print('grad_weight_conv', ctx.needs_input_grad[2])
        # print('grad_bias_conv', ctx.needs_input_grad[3])

        return grad_input_spike, None, grad_weight, grad_bias, None, None

class SYNAPSE_FC(nn.Module):
    def __init__(self, in_features, out_features, trace_const1=1, trace_const2=0.7, TIME=8):
        super(SYNAPSE_FC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.trace_const1 = trace_const1
        self.trace_const2 = trace_const2

        # self.weight = torch.randn(self.out_features, self.in_features, requires_grad=True)
        # self.bias = torch.randn(self.out_features, requires_grad=True)
        self.weight = nn.Parameter(torch.randn(self.out_features, self.in_features))
        self.bias = nn.Parameter(torch.randn(self.out_features))
        # Xavier 초기화
        # nn.init.xavier_uniform_(self.weight)
        # nn.init.constant_(self.bias, 0)

        # ottt
        nn.init.normal_(self.weight, 0, 0.01)
        nn.init.constant_(self.bias, 0)

        self.TIME = TIME

    def forward(self, spike):
        # spike: [Time, Batch, Features]   
        Time = spike.shape[0]
        assert Time == self.TIME, f'Time({Time}) dimension should be same as TIME({self.TIME})'
        Batch = spike.shape[1] 

        # output_current = torch.zeros(Time, Batch, self.out_features, device=spike.device)
        output_current = []

        # spike_detach = spike.detach().clone()
        spike_detach = spike.detach()
        spike_past = torch.zeros_like(spike_detach[0], device=spike.device,requires_grad=False)
        spike_now = torch.zeros_like(spike_detach[0], device=spike.device,requires_grad=False)

        for t in range(Time):
            spike_now = self.trace_const1*spike_detach[t] + self.trace_const2*spike_past
            # output_current[t]= SYNAPSE_FC_METHOD.apply(spike[t], spike_now, self.weight, self.bias) 
            output_current.append( SYNAPSE_FC_METHOD.apply(spike[t], spike_now, self.weight, self.bias) )
            
            spike_past = spike_now

        output_current = torch.stack(output_current, dim=0)
        return output_current 
    



class SYNAPSE_FC_METHOD(torch.autograd.Function):
    @staticmethod
    def forward(ctx, spike_one_time, spike_now, weight, bias):
        ctx.save_for_backward(spike_one_time, spike_now, weight, bias)
        return F.linear(spike_one_time, weight, bias=bias)

    @staticmethod
    def backward(ctx, grad_output_current):
        #############밑에부터 수정해라#######
        spike_one_time, spike_now, weight, bias = ctx.saved_tensors
        
        ## 이거 클론해야되는지 모르겠음!!!!
        grad_output_current_clone = grad_output_current.clone()

        grad_input_spike = grad_weight = grad_bias = None


        if ctx.needs_input_grad[0]:
            grad_input_spike = grad_output_current_clone @ weight
        if ctx.needs_input_grad[2]:
            grad_weight = grad_output_current_clone.t() @ spike_now
        if bias is not None and ctx.needs_input_grad[3]:
            grad_bias = grad_output_current_clone.sum(0)

        # print('grad_input_spike_FC', grad_input_spike)
        # print('grad_weight_FC', grad_weight)
        # print('grad_bias_FC', grad_bias)
        # print('grad_input_spike_FC', ctx.needs_input_grad[0])
        # print('grad_weight_FC', ctx.needs_input_grad[2])
        # print('grad_bias_FC', ctx.needs_input_grad[3])
        
        return grad_input_spike, None, grad_weight, grad_bias

##### First OTTT Synapse ###########################################################
##### First OTTT Synapse ###########################################################
##### First OTTT Synapse ###########################################################






## from NDA paper code #####################################
## from NDA paper code #####################################
## from NDA paper code #####################################
class VGG(nn.Module):

    def __init__(self, cfg, num_classes=10, batch_norm=True, in_c=3,
                lif_layer_v_threshold=0.5, lif_layer_v_decay=0.25, lif_layer_sg_width=1.0):
        super(VGG, self).__init__()

        self.features, out_c = make_layers_nda(cfg, batch_norm, in_c, lif_layer_v_threshold, lif_layer_v_decay, lif_layer_sg_width)
        self.avgpool = SeqToANNContainer(nn.AdaptiveAvgPool2d((1, 1)))
        self.classifier = nn.Sequential(
            SeqToANNContainer(nn.Linear(out_c, num_classes)),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d): #정규화
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
                
    def add_dimension(self, x):
        return add_dimention(x, self.T)
    
    
    def forward(self, x):
        # print('1    ',x.size())
        x = self.add_dim(x) if len(x.shape) == 4 else x
        # print('2      ',x.size())

        x = self.features(x)
        # print('3        ',x.size())
        x = self.avgpool(x)
        # print('4     ',x.size())
        x = torch.flatten(x, 1) if len(x.shape) == 4 else torch.flatten(x, 2)
        # print('5      ',x.size())
        x = self.classifier(x)
        # print('6        ',x.size())
        
        x = x.mean(axis=1)
        # x = x.sum(axis=1)
        return x


def make_layers_nda(cfg, batch_norm=True, in_c=3, lif_layer_v_threshold = 0.5, lif_layer_v_decay = 0.25, lif_layer_sg_width = 1.0):
    layers = []
    in_channels = in_c
    i = 0
    for v in cfg:
        # avgpool이면 H,W절반, conv면 H,W유지.  
        # print('i', i, 'v', v)
        i+=1
        if v == 'P':
            layers += [SpikeModule(nn.AvgPool2d(kernel_size=2, stride=2))]
        elif v == 'M':
            layers += [SpikeModule(nn.MaxPool2d(kernel_size=2, stride=2))]
        else:
            
            layers += [SpikeModule(nn.Conv2d(in_channels, v, kernel_size=3, padding=1))]
            
            # layers += [DimChanger_for_change_0_1()]
            # layers += [SYNAPSE_CONV_BPTT(in_channels=in_channels,
            #                                 out_channels=v, 
            #                                 kernel_size=3, 
            #                                 stride=1, 
            #                                 padding=1, 
            #                                 trace_const1=1, 
            #                                 trace_const2=lif_layer_v_decay,
            #                                 TIME=10)]
            # layers += [DimChanger_for_change_0_1()]
            
            
            

            if batch_norm:
                layers += [tdBatchNorm(v)]
            else:
                pass


            layers += [LIFSpike(lif_layer_v_threshold = 0.5, lif_layer_v_decay = 0.25, lif_layer_sg_width = 1.0)] # 이거 걍 **lif_parameters에 아무것도 없어도 default값으로 알아서 됨.
            
            # layers += [DimChanger_for_change_0_1()]
            # layers += [LIF_layer(v_init=0, 
            #             v_decay=lif_layer_v_decay, 
            #             v_threshold=lif_layer_v_threshold, 
            #             v_reset=0, 
            #             sg_width=lif_layer_sg_width,
            #             surrogate='rough_rectangle',
            #             BPTT_on=True)]
            # layers += [DimChanger_for_change_0_1()]
            

            in_channels = v

    return nn.Sequential(*layers), in_channels



class SeqToANNContainer(nn.Module):
    def __init__(self, *args):
        super().__init__()
        if len(args) == 1:
            self.module = args[0]
        else:
            self.module = nn.Sequential(*args)

    def forward(self, x_seq: torch.Tensor):
        # print('x_seq',x_seq.size())
        y_shape = [x_seq.shape[0], x_seq.shape[1]]
        # print('y_shape',y_shape)
        y_seq = self.module(x_seq.flatten(0, 1).contiguous())
        # print('y_seq',y_seq.size())
        y_shape.extend(y_seq.shape[1:])
        # print('y_shape',y_shape)

        # print('y_seq.view(y_shape)',y_seq.view(y_shape).size())
        return y_seq.view(y_shape)


class SpikeModule(nn.Module):

    def __init__(self, module):
        super().__init__()
        self.ann_module = module

    def forward(self, x):
        B, T, *spatial_dims = x.shape
        a = x.shape
        out = self.ann_module(x.reshape(B * T, *spatial_dims))
        b = out.shape
        
        BT, *spatial_dims = out.shape
        out = out.view(B , T, *spatial_dims).contiguous() # 요소들을 정렬시켜줌.
        return out


def fire_function(gamma):
    class ZIF(torch.autograd.Function): # zero is firing
        @staticmethod
        def forward(ctx, input):
            out = (input >= 0).float()
            # gradient를 위해 input을 저장하는 코드인듯 ㅇㅇ
            # 예의주시해봐
            ctx.save_for_backward(input)
            return out

        @staticmethod
        def backward(ctx, grad_output):
            # forward에서 저장해놨던 input가져오는거임
            (input, ) = ctx.saved_tensors
            grad_input = grad_output.clone()
            tmp = (input.abs() < gamma/2).float() / gamma
            # 사각형 형태의 surrogate gradient임.
            # 1/2 0    ----
            # -1/2 0   |  |
            # 1/2 1    ----
            # -1/2 1
            grad_input = grad_input * tmp
            return grad_input, None

    return ZIF.apply


class LIFSpike(nn.Module):
    def __init__(self, lif_layer_v_threshold = 0.5, lif_layer_v_decay = 0.25, lif_layer_sg_width = 1.0):
        super(LIFSpike, self).__init__()
        self.thresh = lif_layer_v_threshold
        self.tau = lif_layer_v_decay
        self.gamma = lif_layer_sg_width

    def forward(self, x):
        mem = torch.zeros_like(x[:, 0])

        spikes = []
        T = x.shape[1]
        for t in range(T):
            mem = mem * self.tau + x[:, t, ...] #걍 인덱스별로 각각 덧셈
            spike = fire_function(self.gamma)(mem - self.thresh)
            mem = (1 - spike) * mem #spike나감과 동시에 reset
            spikes.append(spike)

        # print('spikes size',spikes.size())
        # print('torch.stack(spikes,dim=1)', torch.stack(spikes, dim=1).size())
            
        # print('xsize22222!!',torch.stack(spikes, dim=1).size())
        
        return torch.stack(spikes, dim=1)



#     tensor.clone()	새롭게 할당	계산 그래프에 계속 상주
# tensor.detach()	공유해서 사용	계산 그래프에 상주하지 않음
# tensor.clone().detach()	새롭게 할당	계산 그래프에 상주하지 않음

def add_dimention(x, T):
    x.unsqueeze_(1)
    x = x.repeat(1, T, 1, 1, 1)
    # T= 10 시계열 데이터 추가
    return x

## from NDA paper code #####################################
## from NDA paper code #####################################
## from NDA paper code #####################################

import numpy as np
from sklearn.metrics import accuracy_score
from scipy.optimize import linear_sum_assignment
from tslearn.clustering import TimeSeriesKMeans
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
from kmodes.kmodes import KModes

def cluster_accuracy(y_true, y_pred):
    """
    Hungarian algorithm을 사용하여 클러스터링 결과의 최적 매칭 후 accuracy 계산
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Confusion Matrix 생성
    D = max(y_pred.max(), y_true.max()) + 1  # 클러스터 개수 결정
    cost_matrix = np.zeros((D, D))
    
    for i in range(D):
        for j in range(D):
            cost_matrix[i, j] = np.sum((y_pred == i) & (y_true == j))
    
    # Hungarian 알고리즘 적용
    row_ind, col_ind = linear_sum_assignment(-cost_matrix)  # Hungarian 알고리즘 (최대화 문제)
    
    # 최적 매핑을 이용하여 y_pred 수정
    mapping = {row: col for row, col in zip(row_ind, col_ind)}
    y_pred_mapped = np.array([mapping[label] for label in y_pred])

    return accuracy_score(y_true, y_pred_mapped)

def evaluate_clustering_accuracy(data, true_labels, n_clusters=3):
    """
    입력 데이터와 라벨을 받아서 3가지 클러스터링 수행 후 accuracy 반환
    """
    n_samples, timesteps, features = data.shape
    
    ##############################################
    # Approach 1: 시계열 DTW 기반 클러스터링
    ##############################################
    dtw_model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", random_state=0)
    labels_dtw = dtw_model.fit_predict(data)

    ##############################################
    # Approach 2: 평탄화 후 Hamming 거리 기반 계층적 클러스터링
    ##############################################
    data_flat = data.reshape(n_samples, -1)  # 50x4 = 200차원 벡터로 평탄화
    distance_matrix = squareform(pdist(data_flat, metric="hamming"))  # Hamming 거리 계산
    agg_cluster = AgglomerativeClustering(n_clusters=n_clusters, affinity="precomputed", linkage="average")
    labels_hamming = agg_cluster.fit_predict(distance_matrix)

    ##############################################
    # Approach 3: k-modes 클러스터링 (범주형/이진 데이터에 적합)
    ##############################################
    km = KModes(n_clusters=n_clusters, init='Huang', n_init=5, verbose=0)
    labels_kmodes = km.fit_predict(data_flat)

    # 정확도 계산
    acc_dtw = cluster_accuracy(true_labels, labels_dtw)
    acc_hamming = cluster_accuracy(true_labels, labels_hamming)
    acc_kmodes = cluster_accuracy(true_labels, labels_kmodes)

    return {
        "DTW Clustering Accuracy": acc_dtw,
        "Hamming Clustering Accuracy": acc_hamming,
        "K-Modes Clustering Accuracy": acc_kmodes
    }
