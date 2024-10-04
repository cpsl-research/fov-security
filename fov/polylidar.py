import numpy as np
from avstack.config import MODELS
from avstack.geometry import GlobalOrigin3D, Polygon
from avstack.sensors import LidarData, ProjectedLidarData
from polylidar import MatrixDouble, Polylidar3D


@MODELS.register_module()
class PolyLidarFovEstimator:
    def __init__(self, lmax: float = 1.8, min_triangles: int = 30):
        self.model = Polylidar3D(lmax=lmax, min_triangles=min_triangles)

    def __call__(self, pc: "LidarData", in_global: bool = False) -> Polygon:
        # run the procedure
        points_mat, pc_bev = self.preprocess(pc, z_min=self.z_min, z_max=self.z_max)
        mesh, polygons = self.execute(self.model, points_mat)
        boundary = self.postprocess(mesh, polygons)

        # package up as a polygon
        fov = Polygon(
            boundary=boundary,
            reference=pc_bev.reference,
            frame=pc.frame,
            timestamp=pc.timestamp,
        )

        # change to global reference if needed
        if in_global:
            fov.change_reference(reference=GlobalOrigin3D, inplace=True)

        return fov

    @staticmethod
    def preprocess(pc: "LidarData", z_min: float, z_max: float) -> MatrixDouble:
        # project into BEV
        if not isinstance(pc, ProjectedLidarData):
            pc_bev = pc.project_to_2d_bev(
                z_min=z_min,
                z_max=z_max,
            )
        else:
            pc_bev = pc

        # run polylidar inference
        points_mat = MatrixDouble(pc_bev.data.x[:, :2], copy=False)
        return points_mat, pc_bev

    @staticmethod
    def execute(model, data: MatrixDouble) -> np.ndarray:
        mesh, _, polygons = model.extract_planes_and_polygons(data)
        return mesh, polygons

    @staticmethod
    def postprocess(mesh, polygons) -> Polygon:
        shell_indices = [polygon.shell for polygon in polygons]
        vertices = np.asarray(mesh.vertices)

        # Assuming there is only one polygon shell for now
        polygon_vertices = [list(vertices[i]) for i in list(shell_indices[0])]
        boundary = np.array(polygon_vertices)

        return boundary
