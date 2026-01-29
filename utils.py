import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import math
from shapely.geometry import Point, Polygon, LineString
import splinecam as sc # This version is a slightly updated version from https://github.com/AhmedImtiazPrio/SplineCAM

def label_points_by_sector(points: np.ndarray, start_angle_deg=90, n_sectors=6):
    angles_rad = np.arctan2(points[:, 1], points[:, 0])
    angles_deg = np.degrees(angles_rad) % 360
    shifted_angles = (angles_deg - start_angle_deg) % 360
    sector_size = 360 / n_sectors
    labels = (shifted_angles // sector_size).astype(int)
    return labels

def generate_polygon(shape='star'):
    if shape=='star':
        cx, cy = (0,0)
        points = []

        for i in range(6):
            angle_deg = i * 60 + 90
            angle_rad = math.radians(angle_deg)
            r = [1.0,1.0,1.0][i//2] if i % 2 == 0 else 0.25
            x = cx + r * math.cos(angle_rad)
            y = cy + r * math.sin(angle_rad)
            points.append((x, y))
        
        points.append(points[0])
        return Polygon(points)
    elif shape=='bowtie':
        points=[(0,0),(np.sqrt(3)/2,-1/np.sqrt(3)),(np.sqrt(3)/2,1/np.sqrt(3)),(0,0),(-np.sqrt(3)/2,1/np.sqrt(3)),(-np.sqrt(3)/2,-1/np.sqrt(3)),(0,0)]
        return Polygon(points)
    elif shape=='reuleaux':
        n_sides=3
        resolution=4
        bulge=1.0
        radius=1.0
        def arc_between(p1, p2):
            p1 = np.array(p1)
            p2 = np.array(p2)
            mid = (p1 + p2) / 2
            direction = p2 - p1
            length = np.linalg.norm(direction)
            ortho = np.array([-direction[1], direction[0]]) / length
            center = mid + bulge * length * ortho
            theta1 = np.arctan2(p1[1] - center[1], p1[0] - center[0])
            theta2 = np.arctan2(p2[1] - center[1], p2[0] - center[0])

            if theta2 < theta1:
                theta2 += 2 * np.pi
            arc_angles = np.linspace(theta1, theta2, resolution)

            arc_points = [
                (center[0] + np.cos(theta) * np.linalg.norm(p1 - center),
                center[1] + np.sin(theta) * np.linalg.norm(p1 - center))
                for theta in arc_angles
            ]
            return arc_points
        angle = 2 * np.pi / n_sides
        vertices = [
            (radius * np.cos(i * angle), radius * np.sin(i * angle))
            for i in range(n_sides)
        ]

        all_points = []
        for i in range(n_sides):
            p1 = vertices[i]
            p2 = vertices[(i + 1) % n_sides]
            arc = arc_between(p1, p2)
            all_points.extend(arc[:-1])

        all_points.append(all_points[0])
        return Polygon(all_points)


def sample_points(polygon, num_inside=100, num_outside=100, xlims=(-2,2), ylims=(-2,2), max_tries=10000):

    inside_points = []
    outside_points = []

    tries = 0
    while (len(inside_points) < num_inside or len(outside_points) < num_outside) and tries < max_tries:
        x = np.random.uniform(xlims[0], xlims[1])
        y = np.random.uniform(ylims[0], ylims[1])
        pt = Point(x, y)
        if polygon.contains(pt):
            if len(inside_points) < num_inside:
                inside_points.append((x, y))
        else:
            if len(outside_points) < num_outside:
                outside_points.append((x, y))
        tries += 1
    inside_points=np.array(inside_points)
    outside_points=np.array(outside_points)
    return inside_points, outside_points

def sample_edges(polygon,num_points):

    ring_coords = list(polygon.exterior.coords)
    edge_segments = [(ring_coords[i], ring_coords[i + 1]) for i in range(len(ring_coords) - 1)]

    edges = [LineString(seg) for seg in edge_segments]
    edge_lengths = [edge.length for edge in edges]
    total_length = sum(edge_lengths)


    n_samples = max(1, num_points)
    spacing = total_length / n_samples

    coords = []
    edge_indices = []

    d = 0  # cumulative distance
    current_edge = 0
    current_edge_start = 0.0

    for i in range(n_samples):
        target_dist = i * spacing

        # Advance to the edge containing the target distance
        while current_edge < len(edges) and current_edge_start + edge_lengths[current_edge] < target_dist:
            current_edge_start += edge_lengths[current_edge]
            current_edge += 1

        if current_edge >= len(edges):
            current_edge = len(edges) - 1  # clamp to last edge

        edge = edges[current_edge]
        edge_dist = target_dist - current_edge_start
        point = edge.interpolate(edge_dist)
        coords.append(tuple(point.coords[0]))
        edge_indices.append(current_edge)
    coords=np.array(coords)
    edge_indices=np.array(edge_indices)
    return coords, edge_indices


def generate_datatset(polygon,**kwargs):

    inside_points,outside_points=sample_points(polygon,**kwargs)

    inside_labels=np.ones(inside_points.shape[0])
    outside_labels=np.zeros(outside_points.shape[0])

    points=np.vstack([inside_points,outside_points])
    labels=np.concatenate([inside_labels,outside_labels])

    points=torch.tensor(points,dtype=torch.float64)
    labels=torch.tensor(labels).long()

    dataset=torch.utils.data.TensorDataset(points,labels)
    return dataset, points, labels

def construct_model(config):
    activation=getattr(nn,config.activation)()
    layers = []
    layers.append(nn.Linear(2, config.width))
    layers.append(activation)

    for _ in range(config.depth - 1):
        layers.append(nn.Linear(config.width, config.width))
        layers.append(activation)

    layers.append(nn.Linear(config.width, 1))
    model=nn.Sequential(*layers)
    model.type(torch.float64)
    return model

def compute_partition(model,xlims=(-2.0,2.0),ylims=(-2.0,2.0)):
    model.eval()
    domain = torch.tensor([
        [xlims[0],ylims[0]],
        [xlims[0],ylims[1]],
        [xlims[1],ylims[1]],
        [xlims[1],ylims[0]],
        [xlims[0],ylims[0]]
    ]).type(torch.float64)

    T = sc.utils.get_proj_mat(domain).type(torch.float64)

    NN=sc.wrappers.model_wrapper(model,input_shape=(2,),T=T,dtype=torch.float64)
    out_cyc,_,Abw,added_edges = sc.compute.get_partitions_with_db(domain,T,NN)

    return out_cyc, Abw, added_edges

def plot_partition(ax,cycles,Abw,colors=None,alpha=0.5,linewidth=0.5):
    cycles = [cyc[:,[1,0]] for cyc in cycles]

    minval,_ = torch.vstack(cycles).min(0)
    maxval,_ = torch.vstack(cycles).max(0)
    if colors is None:
        colors=np.array([m.norm().item() for m in Abw])
        colors-=colors.min()
        colors/=colors.max()
        colors=[plt.cm.jet(c) for c in colors]
    elif colors=='white':
        colors=['white' for _ in Abw]
    elif colors=='random':
        colors=[plt.cm.winter(np.random.random()) for _ in Abw]
    sc.plot.plot_partition(cycles, xlims=[minval[0],maxval[0]],ylims=[minval[1],maxval[1]],alpha=alpha,edgecolor="black",colors=colors,ax=ax, linewidth=linewidth)

def plot_edges(ax,cycles,edges):
    minval,_ = torch.vstack(cycles).min(0)
    maxval,_ = torch.vstack(cycles).max(0)

    for edge in edges:
        ax.plot(edge[:,1],edge[:,0],color='black',alpha=0.75,linewidth=0.5)
    clean_axis(ax)
    ax.set_xlim(minval[1],maxval[1])
    ax.set_ylim(minval[0],maxval[0])


def compute_accuracy(model, loader, device):
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            outputs = model(x).squeeze(1)
            preds = (outputs>0.5).float()#torch.argmax(outputs, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return correct / total if total > 0 else 0.0

def update_model(model,original=nn.ReLU,updated=nn.GELU()):
    new_layers = []
    for layer in model:
        if isinstance(layer, original):
            new_layers.append(updated)
        else:
            new_layers.append(layer)
    return nn.Sequential(*new_layers)

class Centroids: # This class comes from https://github.com/ThomasWalker1/grokalign/tree/main
    def __init__(self, model: nn.Module, device=None):
        self.model = model
        self.device = device if device is not None else next(model.parameters()).device
        self.centroids = None
        self.inner_products = None
        self.norms = None
        self.alignments = None

    def __call__(self, x: torch.Tensor):
        x = x.requires_grad_(True).to(self.device)
        self.centroids = self._compute_centroids(x)
        flat_centroids = self.centroids.reshape(self.centroids.size(0), -1)
        flat_x = x.reshape(x.size(0), -1)

        self.inner_products = (flat_centroids * flat_x).sum(dim=1)

        self.alignments = self.inner_products / torch.clamp(
            flat_centroids.norm(dim=1) * flat_x.norm(dim=1), min=1e-8
        )

        self.norms = flat_centroids.norm(dim=1)

        return self

    def _compute_centroids(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model(x)
        grad_output = torch.ones_like(output)
        centroids = torch.autograd.grad(
            outputs=output,
            inputs=x,
            grad_outputs=grad_output,
            retain_graph=True
        )[0]
        return centroids

    def get_centroids(self) -> torch.Tensor:
        return self.centroids
    
    def get_inner_products(self) -> torch.Tensor:
        return self.inner_products

    def get_norms(self) -> torch.Tensor:
        return self.norms
    
    def get_alignments(self) -> torch.Tensor:
        return self.alignments
    
def remove_neuron(model: nn.Sequential, layer_idx: int, neuron_idx: int) -> nn.Sequential:
    new_layers = []
    skip_next_adjustment = False

    for i, layer in enumerate(model):
        if isinstance(layer, nn.Linear):
            # Copy weights and biases
            weight = layer.weight.data.clone()
            bias = layer.bias.data.clone()

            if i == layer_idx:
                # Remove neuron from this layer's output
                weight = weight[:neuron_idx, :]
                weight = torch.cat((weight, layer.weight.data[neuron_idx+1:, :]), dim=0)
                bias = torch.cat((bias[:neuron_idx], bias[neuron_idx+1:]), dim=0)

                new_layer = nn.Linear(layer.in_features, layer.out_features - 1)
                new_layer.weight.data.copy_(weight)
                new_layer.bias.data.copy_(bias)
                new_layers.append(new_layer)

                skip_next_adjustment = True

            elif skip_next_adjustment:
                # Adjust the next layer's input dimension
                weight = layer.weight.data.clone()
                bias = layer.bias.data.clone()

                weight = torch.cat((weight[:, :neuron_idx], weight[:, neuron_idx+1:]), dim=1)

                new_layer = nn.Linear(layer.in_features - 1, layer.out_features)
                new_layer.weight.data.copy_(weight)
                new_layer.bias.data.copy_(bias)
                new_layers.append(new_layer)

                skip_next_adjustment = False

            else:
                new_layers.append(layer)

        else:
            # Non-linear (e.g., ReLU), just append as-is
            new_layers.append(layer)

    return nn.Sequential(*new_layers)

def neuron_attribution(model,neuron,layer,sample,device):
    model_pruned=remove_neuron(model,layer,neuron).type(torch.float64).to(device)
    ns=[]
    for point in sample:
        point=torch.tensor(point)
        neighborhood=sample_neighborhood(point,0.1,128)
        centroids=Centroids(model)(neighborhood).get_centroids().detach()
        centroids_pruned=Centroids(model_pruned)(neighborhood).get_centroids().detach()
        
        ns.append((centroids_pruned-centroids).square().mean().item()/centroids.norm().item())
    return ns
    
def clean_axis(ax,aspect='auto'):
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_aspect(aspect)

def sample_neighborhood(x: torch.Tensor, radius: float, N: int) -> torch.Tensor:
    d = x.shape[0]
    if N < 1:
        raise ValueError("Number of samples N must be >= 1")
    directions = torch.randn(N - 1, d)
    directions /= directions.norm(dim=1, keepdim=True)
    u = torch.rand(N - 1, 1)
    scales = u.pow(1.0 / d) * radius
    samples = x + scales * directions
    samples = torch.cat([samples, x.unsqueeze(0)], dim=0)
    idx = torch.randperm(N)
    samples = samples[idx]

    return samples

def effective_dimension(X: np.ndarray) -> float:
    X_centered = X - np.mean(X, axis=0, keepdims=True)
    cov = np.cov(X_centered, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.maximum(eigvals, 1e-12)
    p = eigvals / np.sum(eigvals)
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))