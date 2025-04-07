
octree_max_depth = 8
octree_max_gaussians =10
octree_bbox_min = [-1.0, -1.0, -1.0]
octree_bbox_max = [1.0, 1.0, 1.0] 



class OctreeNode:
    def __init__(self, bbox, depth=0, isleaf=True):
        self.bbox = bbox  # 边界框 (min_x, min_y, min_z, max_x, max_y, max_z)
        self.depth = depth
        self.max_depth = octree_max_depth
        self.max_gaussians = octree_max_gaussians
        self.isleaf = isleaf
        self.gaussians = []  # 存储高斯核
        self.children = None  # 子节点

    def split(self):
        """将节点细分为 8 个子节点"""
        if self.depth >= self.max_depth:
            return
        min_x, min_y, min_z, max_x, max_y, max_z = self.bbox
        mid_x = (min_x + max_x) / 2
        mid_y = (min_y + max_y) / 2
        mid_z = (min_z + max_z) / 2
        self.children = [
            OctreeNode((min_x, min_y, min_z, mid_x, mid_y, mid_z), self.depth + 1, self.max_depth, self.max_gaussians),
            OctreeNode((mid_x, min_y, min_z, max_x, mid_y, mid_z), self.depth + 1, self.max_depth, self.max_gaussians),
            OctreeNode((min_x, mid_y, min_z, mid_x, max_y, mid_z), self.depth + 1, self.max_depth, self.max_gaussians),
            OctreeNode((mid_x, mid_y, min_z, max_x, max_y, mid_z), self.depth + 1, self.max_depth, self.max_gaussians),
            OctreeNode((min_x, min_y, mid_z, mid_x, mid_y, max_z), self.depth + 1, self.max_depth, self.max_gaussians),
            OctreeNode((mid_x, min_y, mid_z, max_x, mid_y, max_z), self.depth + 1, self.max_depth, self.max_gaussians),
            OctreeNode((min_x, mid_y, mid_z, mid_x, max_y, max_z), self.depth + 1, self.max_depth, self.max_gaussians),
            OctreeNode((mid_x, mid_y, mid_z, max_x, max_y, max_z), self.depth + 1, self.max_depth, self.max_gaussians),
        ]

    def insert(self, gaussian):
        """将高斯核插入节点"""
        if not self.bbox_contains(gaussian):
            return False
        if self.children is None and len(self.gaussians) < self.max_gaussians:
            self.gaussians.append(gaussian)
            return True
        if self.children is None:
            self.subdivide()
        for child in self.children:
            if child.insert(gaussian):
                return True
        return False

    def bbox_contains(self, gaussian):
        """检查高斯核是否在节点的边界框内"""
        x, y, z = gaussian.get_xyz()
        min_x, min_y, min_z, max_x, max_y, max_z = self.bbox
        return (min_x <= x <= max_x) and (min_y <= y <= max_y) and (min_z <= z <= max_z)
