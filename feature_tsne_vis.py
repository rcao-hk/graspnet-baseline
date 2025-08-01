import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler

data_root = '/media/2TB/result/mmgnet/experiment'
vis_root = 'vis'
os.makedirs(os.path.join(vis_root, 'feature_vis'), exist_ok=True)

geo_feat_root = os.path.join(data_root, 'ignet_v0.6.2')
rgb_feat_root = os.path.join(data_root, 'ignet_v0.8.2.26.2')
camera_type = 'realsense'

perplexity = 50
n_iter = 1000
random_seed = 0

sample_ratio = 0.01
for scene_idx in range(100, 130):
    for anno_idx in range(0, 255, int(1/sample_ratio)):
        geo_feat_path = os.path.join(geo_feat_root, 'scene_{:04d}'.format(scene_idx), camera_type, '{:04d}_middle_feats.npy'.format(anno_idx))
        geo_feat = np.load(geo_feat_path, allow_pickle=True)

        geo_graspness_path = os.path.join(geo_feat_root, 'scene_{:04d}'.format(scene_idx), camera_type, '{:04d}_graspness.npy'.format(anno_idx))
        geo_graspness = np.load(geo_graspness_path, allow_pickle=True)

        rgb_feat_path = os.path.join(rgb_feat_root, 'scene_{:04d}'.format(scene_idx), camera_type, '{:04d}_middle_feats.npy'.format(anno_idx))
        rgb_feat = np.load(rgb_feat_path, allow_pickle=True)

        rgb_graspness_path = os.path.join(rgb_feat_root, 'scene_{:04d}'.format(scene_idx), camera_type, '{:04d}_graspness.npy'.format(anno_idx))
        rgb_graspness = np.load(rgb_graspness_path, allow_pickle=True)

        B, C, N = rgb_feat.shape

        for batch_idx in range(B):
            rgb_inst_feat = rgb_feat[batch_idx]
            geo_inst_feat = geo_feat[batch_idx]

            rgb_inst_graspness = np.mean(rgb_graspness[batch_idx], axis=1)
            geo_inst_graspness = np.mean(geo_graspness[batch_idx], axis=1)


            rgb_inst_feat = rgb_inst_feat.T  # [N, C]
            geo_inst_feat = geo_inst_feat.T  # [N, C]

            # features = np.concatenate([rgb_inst_feat, geo_inst_feat], axis=0)
            # tsne = TSNE(n_components=2, perplexity=50, n_iter=1000, random_state=42)
            # tsne_emb = tsne.fit_transform(features)  # [N, 2]

            # tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=random_seed)
            # tsne_geo_emb = tsne.fit_transform(geo_inst_feat)  # [N, 2]

            # plt.figure(figsize=(8, 6))
            # sc = plt.scatter(tsne_geo_emb[:, 0], tsne_geo_emb[:, 1], s=12, c=geo_inst_graspness, cmap='viridis', alpha=0.8)
            # plt.colorbar(sc, label='Graspness')
            # plt.title('t-SNE Features Colored by Graspness Value')
            # plt.axis('off')
            # plt.tight_layout()
            # plt.savefig(os.path.join(vis_root, 'feature_vis', 'scene_{:04d}_anno_{:04d}_batch_{}_point_graspness.png'.format(scene_idx, anno_idx, batch_idx)), dpi=300)
            # plt.close()

            # tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=random_seed)
            # tsne_rgb_emb = tsne.fit_transform(rgb_inst_feat)  # [N, 2]

            # plt.figure(figsize=(8, 6))
            # sc = plt.scatter(tsne_rgb_emb[:, 0], tsne_rgb_emb[:, 1], s=12, c=rgb_inst_graspness, cmap='viridis', alpha=0.8)
            # plt.colorbar(sc, label='Graspness')
            # plt.title('t-SNE Features Colored by Graspness Value')
            # plt.axis('off')
            # plt.tight_layout()
            # plt.savefig(os.path.join(vis_root, 'feature_vis', 'scene_{:04d}_anno_{:04d}_batch_{}_rgb_graspness.png'.format(scene_idx, anno_idx, batch_idx)), dpi=300)
            # plt.close()
            
            # # # 合并特征，统一进行t-SNE降维
            # combined_features = np.concatenate([rgb_inst_feat, geo_inst_feat], axis=0)

            # tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=random_seed)
            # combined_embedded = tsne.fit_transform(combined_features)

            # N = rgb_inst_feat.shape[0]
            # geo_embedded = combined_embedded[:N]
            # rgb_embedded = combined_embedded[N:]

            # # 可视化
            # plt.figure(figsize=(8, 6))
            # plt.scatter(geo_embedded[:, 0], geo_embedded[:, 1], s=8, c=geo_inst_graspness, alpha=0.6, label='Point-only')
            # plt.scatter(rgb_embedded[:, 0], rgb_embedded[:, 1], s=8, c=rgb_inst_graspness, alpha=0.6, label='RGB Enhanced')
            # plt.title('t-SNE of Point Features: Point-only vs RGB Enhanced')
            # plt.axis('off')
            # plt.legend()
            # # plt.savefig('ap_mean_vs_{}_noise.svg'.format(noise_type), format='svg', dpi=800)
            # plt.savefig(os.path.join(vis_root, 'feature_vis', 'scene_{:04d}_anno_{:04d}_batch_{}.svg'.format(scene_idx, anno_idx, batch_idx)), format='svg', dpi=600)
            # plt.savefig(os.path.join(vis_root, 'feature_vis', 'scene_{:04d}_anno_{:04d}_batch_{}.png'.format(scene_idx, anno_idx, batch_idx)), dpi=600)
            # plt.close()

            geo_scaler = StandardScaler()
            geo_inst_feat_norm = geo_scaler.fit_transform(geo_inst_feat)  # shape [N, C]

            rgb_scaler = StandardScaler()
            rgb_inst_feat_norm = rgb_scaler.fit_transform(rgb_inst_feat)  # shape [N, C]

            tsne_geo = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=random_seed)
            tsne_geo_emb = tsne_geo.fit_transform(geo_inst_feat_norm)  # [N, 2]
            tsne_rgb = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=random_seed)
            tsne_rgb_emb = tsne_rgb.fit_transform(rgb_inst_feat_norm)  # [N, 2]

            # --- 并排可视化 ---
            fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

            sc0 = axes[0].scatter(tsne_geo_emb[:, 0], tsne_geo_emb[:, 1], s=12, c=geo_inst_graspness, cmap='viridis', alpha=0.8)
            axes[0].set_title('Point-only Baseline Feature', fontsize=14, fontweight='bold')
            axes[0].axis('off')
            sc1 = axes[1].scatter(tsne_rgb_emb[:, 0], tsne_rgb_emb[:, 1], s=12, c=rgb_inst_graspness, cmap='viridis', alpha=0.8)
            axes[1].set_title('MMGNet Feature', fontsize=14, fontweight='bold')
            axes[1].axis('off')

            # colorbar共用，放右侧
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes
            cax = inset_axes(axes[1], width="4%", height="100%", loc='lower left',
                            bbox_to_anchor=(1.05, 0., 1, 1), bbox_transform=axes[1].transAxes, borderpad=0)
            cb = fig.colorbar(sc1, cax=cax)
            cb.ax.tick_params(labelsize=10)

            # fig.colorbar(sc1, ax=axes.ravel().tolist(), label='Graspness', shrink=0.7, pad=0.03)
            # plt.tight_layout()
            plt.subplots_adjust(right=0.85)  # 让整体更紧凑
            plt.savefig(os.path.join(
                vis_root, 'feature_vis', 
                f'scene_{scene_idx:04d}_anno_{anno_idx:04d}_batch_{batch_idx}_tsne_sidebyside.png'
            ), dpi=600)
            plt.savefig(os.path.join(
                vis_root, 'feature_vis', 
                f'scene_{scene_idx:04d}_anno_{anno_idx:04d}_batch_{batch_idx}_tsne_sidebyside.svg'
            ), format='svg', dpi=600)
            plt.close()