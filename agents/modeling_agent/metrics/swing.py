import numpy as np
from typing import Dict, List, Tuple

class SwingMetrics:
    """バッティングスイングの分析メトリクスを計算するクラス"""
    
    def detect_swing_phases(self, keypoints_3d: np.ndarray) -> Dict[str, int]:
        """スイングの各フェーズの開始フレームを検出する"""
        phases = {
            "stance": 0,
            "load": 0,
            "stride": 0,
            "contact": 0,
            "follow_through": 0
        }
        
        # バットと体の動きのデータを取得
        bat_movement = self._calculate_bat_movement(keypoints_3d)
        body_rotation = self._calculate_body_rotation(keypoints_3d)
        
        # フェーズの検出
        phases["load"] = self._detect_load_phase(bat_movement, body_rotation)
        phases["stride"] = self._detect_stride_phase(keypoints_3d)
        phases["contact"] = self._detect_contact_phase(bat_movement)
        phases["follow_through"] = self._detect_follow_through(bat_movement)
        
        return phases

    def calculate_bat_speed(self, keypoints_3d: np.ndarray, contact_frame: int) -> float:
        """インパクト付近のバットスピードを計算"""
        # バットのヘッド部分のキーポイントを特定
        bat_head = keypoints_3d[:, -1, :]  # 最後のキーポイントがバットヘッド想定
        
        # インパクト前後のフレームでの速度を計算
        window = 5  # インパクト前後のフレーム数
        speeds = []
        
        for i in range(max(0, contact_frame - window), min(len(keypoints_3d), contact_frame + window)):
            if i > 0:
                velocity = bat_head[i] - bat_head[i-1]
                speed = np.linalg.norm(velocity)
                speeds.append(speed)
        
        # 最大スピードを返す
        return max(speeds) if speeds else 0.0

    def calculate_rotation_speed(self, keypoints_3d: np.ndarray, part: str) -> float:
        """体の回転スピードを計算"""
        if part == "hips":
            joint_indices = [11, 12]  # 左右の腰のインデックス
        elif part == "shoulders":
            joint_indices = [5, 6]  # 左右の肩のインデックス
        else:
            raise ValueError(f"Unknown body part: {part}")
        
        # 各フレームでの回転角度を計算
        angles = []
        for frame in keypoints_3d:
            vector = frame[joint_indices[1]] - frame[joint_indices[0]]
            angle = np.arctan2(vector[2], vector[0])  # XZ平面での角度
            angles.append(angle)
        
        # 角速度を計算（フレーム間の角度の変化）
        angular_velocities = np.diff(angles)
        
        # 最大角速度を返す
        return max(abs(angular_velocities))

    def evaluate_rotation_sequence(
        self,
        keypoints_3d: np.ndarray,
        phases: Dict[str, int]
    ) -> float:
        """回転の連動性（キネマティックチェーン）を評価"""
        # 各部位の回転タイミングを取得
        hip_rotation = self._get_rotation_timing("hips", keypoints_3d)
        shoulder_rotation = self._get_rotation_timing("shoulders", keypoints_3d)
        arms_rotation = self._get_rotation_timing("arms", keypoints_3d)
        
        # 理想的な順序（腰→肩→腕）からのずれを計算
        ideal_sequence = sorted([hip_rotation, shoulder_rotation, arms_rotation])
        actual_sequence = [hip_rotation, shoulder_rotation, arms_rotation]
        
        sequence_score = 1.0
        for ideal, actual in zip(ideal_sequence, actual_sequence):
            if ideal != actual:
                sequence_score *= 0.8  # ペナルティ
        
        return sequence_score

    def analyze_weight_shift(
        self,
        keypoints_3d: np.ndarray,
        phases: Dict[str, int]
    ) -> float:
        """重心移動の効率性を分析"""
        # 腰のキーポイントから重心位置を推定
        center_of_mass = self._calculate_center_of_mass(keypoints_3d)
        
        # ストライド開始から接触までの横方向の重心移動を計算
        start_frame = phases["stride"]
        end_frame = phases["contact"]
        
        lateral_movement = center_of_mass[end_frame, 0] - center_of_mass[start_frame, 0]
        movement_smoothness = self._calculate_movement_smoothness(
            center_of_mass[start_frame:end_frame+1]
        )
        
        # 移動距離と滑らかさを組み合わせてスコア化
        weight_shift_score = (lateral_movement * 0.6 + movement_smoothness * 0.4)
        
        return weight_shift_score

    def calculate_swing_plane(self, keypoints_3d: np.ndarray) -> float:
        """スイング軌道の平面性を計算"""
        # バットヘッドの軌跡を取得
        bat_head_trajectory = keypoints_3d[:, -1, :]  # 最後のキーポイントがバットヘッド
        
        # 主成分分析で平面を特定
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3)
        pca.fit(bat_head_trajectory)
        
        # 最小の主成分の寄与率が小さいほど平面的
        plane_score = 1 - pca.explained_variance_ratio_[2]
        
        return plane_score

    # プライベートヘルパーメソッド
    def _calculate_bat_movement(self, keypoints_3d: np.ndarray) -> np.ndarray:
        """バットの動きを計算"""
        bat_head = keypoints_3d[:, -1, :]
        return np.diff(bat_head, axis=0)

    def _calculate_body_rotation(self, keypoints_3d: np.ndarray) -> np.ndarray:
        """体の回転を計算"""
        shoulders = keypoints_3d[:, 5:7, :]  # 肩のキーポイント
        shoulder_vectors = shoulders[:, 1] - shoulders[:, 0]
        return np.arctan2(shoulder_vectors[:, 2], shoulder_vectors[:, 0])

    def _detect_load_phase(
        self,
        bat_movement: np.ndarray,
        body_rotation: np.ndarray
    ) -> int:
        """ロード相の開始フレームを検出"""
        # 体の回転が逆方向に始まるポイントを検出
        rotation_change = np.where(np.diff(body_rotation) < -0.1)[0]
        if len(rotation_change) > 0:
            return rotation_change[0]
        return 0

    def _detect_stride_phase(self, keypoints_3d: np.ndarray) -> int:
        """ストライド相の開始フレームを検出"""
        # 前足の動き出しを検出
        front_foot = keypoints_3d[:, 3, :]  # 前足のキーポイント
        foot_movement = np.diff(front_foot[:, 0])  # X方向の動き
        movement_start = np.where(abs(foot_movement) > 0.05)[0]
        if len(movement_start) > 0:
            return movement_start[0]
        return 0

    def _detect_contact_phase(self, bat_movement: np.ndarray) -> int:
        """インパクト時のフレームを検出"""
        # バットスピードが最大になるポイント
        bat_speed = np.linalg.norm(bat_movement, axis=1)
        return np.argmax(bat_speed) + 1

    def _detect_follow_through(self, bat_movement: np.ndarray) -> int:
        """フォロースルー開始フレームを検出"""
        bat_speed = np.linalg.norm(bat_movement, axis=1)
        # スピードが落ち始めるポイント
        speed_decrease = np.where(np.diff(bat_speed) < -0.1)[0]
        if len(speed_decrease) > 0:
            return speed_decrease[0] + 1
        return len(bat_movement)

    def _get_rotation_timing(self, part: str, keypoints_3d: np.ndarray) -> int:
        """各部位の回転開始タイミングを取得"""
        if part == "hips":
            joints = [11, 12]
        elif part == "shoulders":
            joints = [5, 6]
        elif part == "arms":
            joints = [6, 7]
        else:
            raise ValueError(f"Unknown body part: {part}")
            
        vectors = keypoints_3d[:, joints[1]] - keypoints_3d[:, joints[0]]
        angles = np.arctan2(vectors[:, 2], vectors[:, 0])
        
        # 角速度が一定以上になるフレームを検出
        angular_velocity = np.diff(angles)
        return np.where(abs(angular_velocity) > 0.1)[0][0]

    def _calculate_center_of_mass(self, keypoints_3d: np.ndarray) -> np.ndarray:
        """重心位置の推定"""
        # 簡易的な重み付け
        weights = {
            'pelvis': 0.35,
            'torso': 0.35,
            'head': 0.1,
            'arms': 0.1,
            'legs': 0.1
        }
        
        com = np.zeros_like(keypoints_3d[:, 0, :])
        
        # 骨盤
        com += weights['pelvis'] * keypoints_3d[:, 0, :]
        
        # 胴体
        torso = (keypoints_3d[:, 7, :] + keypoints_3d[:, 8, :]) / 2
        com += weights['torso'] * torso
        
        # 頭
        com += weights['head'] * keypoints_3d[:, 10, :]
        
        # 腕（簡易的に）
        arms = (keypoints_3d[:, 5, :] + keypoints_3d[:, 6, :]) / 2
        com += weights['arms'] * arms
        
        # 脚（簡易的に）
        legs = (keypoints_3d[:, 1, :] + keypoints_3d[:, 2, :]) / 2
        com += weights['legs'] * legs
        
        return com

    def _calculate_movement_smoothness(self, trajectory: np.ndarray) -> float:
        """動きの滑らかさを計算"""
        # 速度の変化率（加速度）を計算
        velocities = np.diff(trajectory, axis=0)
        accelerations = np.diff(velocities, axis=0)
        
        # 加速度の変化が小さいほど滑らか
        smoothness = 1 / (1 + np.mean(np.linalg.norm(accelerations, axis=1)))
        
        return smoothness