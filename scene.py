import math
import random

import cv2
import numpy as np

from panda3d.bullet import BulletRigidBodyNode
from panda3d.bullet import BulletTriangleMeshShape, BulletHeightfieldShape, ZUp
from panda3d.bullet import BulletConvexHullShape, BulletTriangleMesh
from panda3d.core import NodePath, PandaNode
from panda3d.core import BitMask32, Vec3, Point3, LColor
from panda3d.core import Filename, PNMImage, Texture
from panda3d.core import GeoMipTerrain
from panda3d.core import Shader, TextureStage, TransformState
from panda3d.core import TransparencyAttrib
from panda3d.core import OmniBoundingVolume, Quat
# from shapes import Cylinder, Plane
# from lights import BasicAmbientLight, BasicDayLight
from panda3d.core import Shader, ShaderBuffer, GeomEnums

from voronoi_generator.voronoi_2d import Polygon2DMixin
from voronoi_generator.voronoi_2d import VoronoiSitesGenerator, RoundedVoronoiGenerator, BoundedVoronoiGenerator
from texture_generator.heightmap_generator import RoundedVoronoiRockPillarsHeightmap
from texture_generator.noise_compositing import SimplexNoiseBlending

from enum import StrEnum

import cv2
import numpy as np
from panda3d.bullet import BulletHeightfieldShape, ZUp

from panda3d.core import SamplerState, TransformState


from texture_generator.heightmap_generator import ProcessHeightmapMixin
from shapes import RandomPolygonalPrism


class ImageAssets(StrEnum):

    @classmethod
    def iter_load(cls):
        for mem in cls:
            tex = base.loader.load_texture(f'textures/{mem.value}')
            tex_name = mem.name.lower()
            yield (tex_name, tex)


class BottomIslandTextures(ImageAssets):

    # TEX_GROUND = 'ground_01.jpg'
    # TEX_ROCK1 = 'tex_rock.png'
    # TEX_ROCK2 = "tex_cracked.png"
    # TEX_ROCK3 = "tex_moss.png"

    TEX_GROUND = 'grass_01.jpg'
    TEX_ROCK1 = 'rock_02_512.jpg'
    TEX_ROCK2 = "matched_texture_hsv_robust_20261105131229.png"
    TEX_ROCK3 = "matched_texture_lab_20261105130815.jpg"



class Sky(NodePath):

    def __init__(self):
        super().__init__(PandaNode('sky'))
        model = base.loader.load_model('models/blue-sky/blue-sky-sphere')
        model.set_color(LColor(2, 2, 2, 1))
        model.set_scale(0.2)
        model.set_z(0)
        model.reparent_to(self)
        self.set_shader_off()


class Terrain(NodePath):

    def __init__(self, heightmap_path=None, texmap_path=None, height=100):
        super().__init__(BulletRigidBodyNode('terrain'))
        self.height = height
        self.node().set_mass(0)
        self.set_collide_mask(BitMask32.bit(1))

        self.generate_terrain(heightmap_path, texmap_path)
        # self.create_terrain(heightmap_path)
        # self.setup_shader(texmap_path)

    def generate_terrain(self, heightmap_path, texmap_path):
        heightmap_img = texmap_img = None

        if heightmap_path and texmap_path:
            if (heightmap_img := cv2.imread(heightmap_path)) is None:
                print(f'Heightmap not found or could not be read: {heightmap_path}')

            if (texmap_img := cv2.imread(texmap_path)) is None:
                print(f'Texmap not found or could not be read: {texmap_path}')

        if heightmap_img is None or texmap_img is None:
            t = random.uniform(0, 1000)

            maker = RoundedVoronoiRockPillarsHeightmap(grid=3, size=257, blur=None, tp=10, t=t, random_height=True)
            arr = maker.create_heightmap(bit=8)
            blender = SimplexNoiseBlending(debug=False, noise_scale=100, noise_intensity=100)
            heightmap_arr = blender.blend(arr)
            heightmap_arr = np.stack([heightmap_arr] * 3, axis=-1)

            maker = RoundedVoronoiRockPillarsHeightmap(grid=3, size=257, blur=None, tp=10, t=t, random_height=False)
            texmap_arr = maker.create_heightmap(bit=8)
            texmap_arr = np.stack([texmap_arr] * 3, axis=-1)

        self.create_terrain(heightmap_arr)
        self.setup_shader(texmap_arr)

        cv2.imwrite('sample_heightmap.png', heightmap_arr)
        cv2.imwrite('sample_texmap.png', texmap_arr)

        return heightmap_img

    def generate_texture_from_img(self, img, component_type=Texture.T_unsigned_byte,
                                  format=Texture.F_rgb):
        """Dynamically create a panda3d.core.Texture from a Numpy.ndarray.
            Args:
                img (Numpy.ndarray): Texture image.
                component_type (int): Component type like Texture.T_unsigned_byte
                format (int): Format like Texture.F_rgb

        """
        tex = Texture('image')
        size = img.shape[0]

        tex.setup_2d_texture(
            x_size=size,
            y_size=size,
            component_type=component_type,
            format=format
        )

        tex.set_ram_image(img)
        mem_view = memoryview(tex.modify_ram_image())
        tex.set_ram_image(mem_view)

        return tex

    # def create_terrain(self, heightmap_path):
    def create_terrain(self, img):
        # img = cv2.imread(heightmap_path)
        img = cv2.GaussianBlur(img, (11, 11), 0)
        tex_heightfield = self.generate_texture_from_img(img)
        # Convert the texture to PNMImage.
        pnm_img = PNMImage()
        tex_heightfield.store(pnm_img)


        # img = PNMImage(Filename(heightmap_path))
        # shape = BulletHeightfieldShape(img, self.height, ZUp)
        shape = BulletHeightfieldShape(pnm_img, self.height, ZUp)
        shape.set_use_diamond_subdivision(True)
        self.node().add_shape(shape)

        self.terrain = GeoMipTerrain('geomip_terrain')
        # self.terrain.set_heightfield(heightmap_path)
        self.terrain.set_heightfield(pnm_img)
        self.terrain.set_border_stitching(True)

        self.terrain.set_block_size(8)
        self.terrain.set_min_level(2)
        self.terrain.set_focal_point(base.camera)

        # size_x, size_y = img.get_size()
        self.heightmap_size = pnm_img.get_size()
        x = (self.heightmap_size.x - 1) / 2
        y = (self.heightmap_size.y - 1) / 2

        pos = Point3(-x, -y, -(self.height / 2))
        self.terrain_scale = Vec3(1, 1, self.height)
        self.root = self.terrain.get_root()
        self.root.set_scale(self.terrain_scale)
        self.root.set_pos(pos)
        self.terrain.generate()
        self.root.reparent_to(self)

    # def setup_shader(self, texmap_path):
    def setup_shader(self, img):
        # img = cv2.imread(texmap_path)
        img = cv2.GaussianBlur(img, (11, 11), 0)
        texmap = self.generate_texture_from_img(img)
        # Convert the texture to PNMImage.
        shader = Shader.load(Shader.SL_GLSL, 'shaders/terrain_v.glsl', 'shaders/terrain_f.glsl')
        self.root.set_shader(shader)

        self.root.set_shader_input("height_scale", self.height)
        self.root.set_shader_input("terrain_size", 256)
        # self.root.set_shader_input('tex_heightfield', base.loader.load_texture(texmap_path))
        # self.root.set_shader_input('tex_mask', base.loader.load_texture(texmap_path))
        self.root.set_shader_input('tex_mask', texmap)

        for tex_name, tex in BottomIslandTextures.iter_load():
            tex.setMinfilter(SamplerState.FT_linear_mipmap_linear)
            tex.set_anisotropic_degree(16)
            tex.wrap_u = SamplerState.WM_repeat
            tex.wrap_v = SamplerState.WM_repeat
            self.root.set_shader_input(tex_name, tex)

    def calc_normal_from_heightmap(self, px, py):
        """Calculate the exact normals from the pixel coordinates
           (indices of a numpy.ndarray) in a heightmap image.

            Args:
                px, py(int): Indices of a numpy.ndarray(heightmap image)
            Returns:
                LVector3f: Normalized normal

        """
        sx, sy, sz = self.terrain_scale
        w, h = self.heightmap_size

        # Limit the top, bottom, left, and right indexes so they do not extend beyond the boundaries.
        px0 = max(0, px - 1)
        px1 = min(w - 1, px + 1)
        py0 = max(0, py - 1)
        py1 = min(h - 1, py + 1)

        # Apply the scale sz and retrieve the height (Z) in actual world coordinates.
        z_left = self.terrain.get_elevation(px0, py) * sz
        z_right = self.terrain.get_elevation(px1, py) * sz
        z_down = self.terrain.get_elevation(px, py0) * sz
        z_up = self.terrain.get_elevation(px, py1) * sz

        # Distance in the X-direction and distance in the Y-direction.
        dx = (px1 - px0) * sx
        dy = (py1 - py0) * sy

        # Calculate the cross product of the tangent vectors in the X-axis
        # and the Y-axis directions to find the exact normal vector.
        tangent_x = Vec3(dx, 0, z_right - z_left)
        tangent_y = Vec3(0, dy, z_up - z_down)
        normal = tangent_x.cross(tangent_y)
        normal.normalize()
        return normal

    def get_terrain_z(self, px, py):
        """Retrieve terrain elevation from the pixel coordinates (indices of a numpy.ndarray)
           of a height map image.

            Args:
                px, py(int): Indices of a numpy.ndarray(heightmap image)
        """
        raw_z = self.terrain.get_elevation(px, py)
        z = raw_z * self.terrain_scale.z - self.terrain_scale.z / 2
        return z



class Building(Polygon2DMixin, NodePath):

    def __init__(self, name, foundation_h=15, wall_h=10, belt_h=2):
        super().__init__(BulletRigidBodyNode(name))
        self.set_collide_mask(BitMask32.bit(1))
        self.node().set_mass(0)

        self.foundation_h = foundation_h
        self.wall_h = wall_h
        self.belt_h = belt_h

    def calc_tex_scale(self, total_length, distance):
        """Calculate texture scale.
            Args:
                total_length (float): Total distance
                distance (float):
                    Specify the distance in 3D space at which the texture repeats once (one cycle).
        """
        repeats = total_length / distance

        if (s := round(repeats)) < 1:
            s = 1

        return s

    def assemble(self, model, pos, name):
        shape = BulletConvexHullShape()
        shape.add_geom(model.node().get_geom(0))

        self.node().add_shape(shape, TransformState.make_pos(pos))
        model.set_pos(pos)
        model.set_name(name)
        model.reparent_to(self)

    def get_terrain_z(self, world_cx, world_cy):
        from_pos = Point3(world_cx, world_cy, 150)
        to_pos = Point3(world_cx, world_cy, 0)  # 5

        if (result := base.world.ray_test_closest(
                from_pos, to_pos, BitMask32.bit(1))).has_hit():
            return result.get_hit_pos()

    def generate(self, vertices, textures, offset):
        # Determine the value of segs_top_cap based on the vertex farthest from the center.
        _, max_distance = self.get_max_distance_from_center(vertices)
        segs_cap = 3 if max_distance <= 2 else int(max_distance / 2)

        wall_creator = RandomPolygonalPrism(list(vertices), segs_top_cap=0, segs_bottom_cap=0)
        belt_creator = RandomPolygonalPrism(list(vertices * 1.02), height=self.belt_h, segs_top_cap=segs_cap, segs_bottom_cap=segs_cap)

        foundation_pos = Point3(*wall_creator.center) + offset

        # ##############################
        hit_pos = self.get_terrain_z(foundation_pos.x, foundation_pos.y)
        foundation_pos.z = hit_pos.z - 10

        # ##############################
        

        self.create_foundation(foundation_pos, wall_creator, textures)

        wall_pos = foundation_pos + Vec3(0, 0, self.foundation_h)
        self.create_wall(wall_pos, wall_creator, belt_creator, textures)

    def create_foundation(self, pos, wall_creator, textures):
        """Create building foundation.
        """
        # create_model.
        wall_creator.height = self.foundation_h
        model = wall_creator.create()

        # set texture.
        su = self.calc_tex_scale(wall_creator.edge_length, 25)
        sv = self.calc_tex_scale(self.foundation_h, 25)
        tex = textures['foundation']
        model.set_tex_scale(TextureStage.get_default(), (su, sv))
        model.set_texture(tex)

        self.assemble(model, pos, name="foundtion")

    def create_wall(self, pos, wall_creator, belt_creator, textures):
        """Create building wall.
        """
        tex_belt = textures['belt']
        tex_walls = textures['walls']

        wall_creator.height = self.wall_h
        wall_creator.segs_a = int(self.wall_h / 2)

        for i, tex_wall in enumerate(tex_walls):
            # create wall.
            wall = wall_creator.create()
            su = self.calc_tex_scale(wall_creator.edge_length, 20)
            wall.set_tex_scale(TextureStage.get_default(), (su, 1))
            wall.set_texture(tex_wall)
            self.assemble(wall, pos, f'wall{i}')
            pos.z += wall_creator.height

            # Create belt between walls.
            belt = belt_creator.create()
            belt.set_texture(tex_belt)
            self.assemble(belt, pos, f'belt{i}')
            pos.z += belt_creator.height


class TownBuilder(ProcessHeightmapMixin, Polygon2DMixin):

    def __init__(self, file_path, terrain, area_ratio=0.75):
        # arr = cv2.imread(file_path)
        # arr = self.create_bw_texture_map(arr, blur=None)
        # cv2.imwrite('sample.png', arr)
        self.file_path = file_path
        self.area_ratio = area_ratio

        self.terrain = terrain

        self.create_textures()

    def create_textures(self):
        self.tex_building = {
            'foundation': self.load_tex('brick_04.jpg'),
            'belt': self.load_tex('concrete_09.jpg'),
            'walls': None
        }

        self.tex_walls = [
            self.load_tex('wall_window1.png'),  # most decorative
            self.load_tex('wall_window2.png'),  # a little decorative
            self.load_tex('wall_window3.png'),  # square
        ]

    def load_tex(self, file_name):
        return base.loader.load_texture(f'textures/{file_name}')

    def get_terrain_z(self, world_cx, world_cy):
        from_pos = Point3(world_cx, world_cy, 150)
        to_pos = Point3(world_cx, world_cy, 5)

        if (result := base.world.ray_test_closest(
                from_pos, to_pos, BitMask32.bit(1))).has_hit():
            return result.get_hit_pos()

    def create_building(self, num, img_size, sorted_pts, offset_pos):
        stories = random.randint(1, 3)
        self.tex_building['walls'] = self.tex_walls[:1] * stories if stories <= 2 \
            else [self.tex_walls[1]] * (stories - 1) + [self.tex_walls[2]]

        building = Building(f'building_{num}')
        scaled_pts = sorted_pts * img_size * self.area_ratio
        building.generate(scaled_pts, self.tex_building, offset_pos)
        return building

    def __iter__(self):
        img = cv2.imread(self.file_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.flip(img, 0)
        h, w = img.shape[:2]
        offset_cx, offset_cy = w / 2.0, h / 2.0

        # ###########################
        vertices_li = []
        # ###########################

        for i, (center, vertices) in enumerate(self.generate_contour_vertices(img)):
            # ###########################
            if i % 2 == 1:
                shrinked = ((vertices - center) * 0.6 + center).astype(np.int32)
                vertices_li.append(shrinked)
                continue
            # ###########################

            world_cx = center[0] - offset_cx
            world_cy = offset_cy - center[1]

            local_vertices = vertices - center
            local_vertices[:, 1] = -local_vertices[:, 1]

            # if not (hit_pos := self.get_terrain_z(world_cx, world_cy)):
            #     continue

            region = (local_vertices / h) * self.area_ratio
            sites = np.array([pt for pt in VoronoiSitesGenerator(region)])
            # offset_pos = Point3(world_cx, world_cy, hit_pos.z)
            offset_pos = Point3(world_cx, world_cy, 0)

            # Generate vertex coordinates of a rounded-corner Voronoi cell.
            for i, pts in enumerate(RoundedVoronoiGenerator(
                    pts=sites, bnd=region, segment_length=0.0029)):

                if len(pts) == 0:
                    continue

                polygon = np.insert(pts, pts.shape[1], 0, axis=1)
                sorted_pts = self.sort_counter_clockwise(polygon)

                building = self.create_building(i, h, sorted_pts, offset_pos)
                yield building

        # ###########################
        vegetation_map = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(vegetation_map, vertices_li, -1, color=255, thickness=cv2.FILLED)
        vegetation_map = cv2.flip(vegetation_map, 0)

        result_map = np.zeros((h, w), dtype=np.uint8)
        mask = vegetation_map == 255

        # # # vegetation_indices = np.vstack(vertices_li)
        org_image_path = 'terrain/8bit_SimplexNoiseBlending_20261123102857.png'
        org_image = cv2.imread(org_image_path, cv2.IMREAD_GRAYSCALE)

        result_map[mask] = org_image[mask]
        self.vegetation(result_map, world_cx, world_cy)
        # cv2.imwrite('vegetation_map.png', result_map)

        # final_vegetation_map = np.zeros((h, w), dtype=np.uint8)
        # final_vegetation_map[coords_xy] = org_image[coords_xy]
        # cv2.imwrite('vegetation_map.png', final_vegetation_map)

        # vegetation_map = np.zeros((h, w), dtype=np.uint8)
        # cv2.drawContours(vegetation_map, vertices_li, -1, color=255, thickness=cv2.FILLED)
        # vegetation_map = cv2.flip(vegetation_map, 0)
        # cv2.imwrite('vegetation_map.png', vegetation_map)
        # self.vegetation(vegetation_map)
        # ###########################

    def vegetation(self, vegetation_map, *_):
        h, w = vegetation_map.shape[:2]
        threshold = 0.5  # 0.85
        num_samples = 300

        v_min = vegetation_map.min()
        v_max = vegetation_map.max()
        vegetation_map = (vegetation_map - v_min) / (v_max - v_min)

        coords = np.argwhere(vegetation_map >= threshold)

        if len(coords) > 0:
            dummy = NodePath('dummy_transform')
            mat_plants = []
            
            sample_size = min(len(coords), num_samples)
            indices = np.random.choice(len(coords), size=sample_size, replace=False)
            selected_coords = coords[indices]

            sx = self.terrain.root.get_sx()
            sy = self.terrain.root.get_sy()
            sz = self.terrain.root.get_sz()

            for y, x in selected_coords:
                # 端のピクセルではみ出さないように制限
                # x0 = max(0, x - 1)
                # x1 = min(w - 1, x + 1)
                # y0 = max(0, y - 1)
                # y1 = min(h - 1, y + 1)

                # # 実際のワールド座標での高さ(Z)を取得（スケール sz を適用済み）
                # z_left  = self.terrain.terrain.get_elevation(x0, y) * sz
                # z_right = self.terrain.terrain.get_elevation(x1, y) * sz
                # z_down  = self.terrain.terrain.get_elevation(x, y0) * sz
                # z_up    = self.terrain.terrain.get_elevation(x, y1) * sz

                # # 実際のワールド空間でのX方向の距離とY方向の距離
                # dx = (x1 - x0) * sx
                # dy = (y1 - y0) * sy

                # # X軸方向の接線ベクトルと、Y軸方向の接線ベクトルの外積 ＝ 正確な法線！
                # tangent_x = Vec3(dx, 0, z_right - z_left)
                # tangent_y = Vec3(0, dy, z_up - z_down)
                # normal = tangent_x.cross(tangent_y)
                # normal.normalize()
                normal = self.terrain.calc_normal_from_heightmap(x, y)

                # 植物の配置座標
                # raw_z = self.terrain.terrain.get_elevation(x, y)
                # z = raw_z * sz - 50
                z = self.terrain.get_terrain_z(x, y)
                pos = Point3(x - 257 / 2, y - 257 / 2, z)

                mat_data = self.transform_plant(dummy, pos, normal)
                mat_plants.append(mat_data)

            self.create_model(mat_plants)



    def vegetation_1(self, vegetation_map, world_x, world_y):
        threshold = 0.9
        num_samples = 200

        v_min = vegetation_map.min()
        v_max = vegetation_map.max()
        vegetation_map = (vegetation_map - v_min) / (v_max - v_min)

        coords = np.argwhere(vegetation_map >= threshold)

        if len(coords) > 0:
            dummy = NodePath('dummy_transform')
            mat_plants = []
            
            sample_size = min(len(coords), num_samples)
            indices = np.random.choice(len(coords), size=sample_size, replace=False)
            selected_coords = coords[indices]

            for y, x in selected_coords:
                normal = self.terrain.terrain.get_normal(x, y)

                normal.set(
                    normal.x / self.terrain.root.get_sx(),
                    -normal.y / self.terrain.root.get_sy(),
                    normal.z / self.terrain.root.get_sz()
                )
                normal.normalize()

                raw_z = self.terrain.terrain.get_elevation(x, y)
                z = raw_z * self.terrain.root.get_sz() - 50
                # vegetation_map
                # pos = Point3(x - 257 / 2, y - 257/2, z)
                pos = Point3(x - 257 / 2, y - 257 / 2, z)
                mat_data = self.transform_plant(dummy, pos, normal)
                mat_plants.append(mat_data)

            self.create_model(mat_plants)
            # mat_plants = []

            # for coords in selected_coords:


        # geomip_terrainからnormalを取得
        # # ピクセル座標 (x, y) の法線を取得
        # normal = terrain.get_normal(x, y)

        # # 【重要】get_normalは地形のスケールを考慮しないため、スケールで割って再正規化する
        # normal.set(
        #     normal.x / root.get_sx(),
        #     normal.y / root.get_sy(),
        #     normal.z / root.get_sz()
        # )
        # normal.normalize()


    def create_model(self, matrices, color_scale=None):
        arr_matrices = np.array(matrices, dtype=np.float32)
        raw_buffer_data = arr_matrices.tobytes()
        # model = base.loader.load_model('models/plants1/plants1.egg')
        model = base.loader.load_model('models/shrubbery/shrubbery.egg')

        if color_scale is not None:
            # Set the color to white first, since the color of plant1 didn't change with just set_color_scale.
            model.set_color(1, 1, 1, 1)
            model.set_color_scale(*color_scale, 1)

        model.flatten_light()
        model.reparent_to(base.render)
        model.set_pos(0, 0, 0)
        model.set_hpr(0, 0, 0)

        # Prevent curling.
        for geom_np in model.find_all_matches("**/+GeomNode"):
            geom_nd = geom_np.node()
            geom_nd.set_bounds(OmniBoundingVolume())
            geom_nd.set_final(True)

        # model.node().set_bounds(OmniBoundingVolume())
        # model.node().set_final(True)

        # Set the number of instances.
        instance_cnt = len(matrices)
        model.set_instance_count(instance_cnt)
        # Set shader.
        model.set_shader(Shader.load(Shader.SL_GLSL, vertex='shaders/instancing_v.glsl', fragment='shaders/instancing_f.glsl'))
        model.set_shader_input("instanced_object", ShaderBuffer('DataBuffer', raw_buffer_data, GeomEnums.UH_static))

    def calc_hpr(self, normal):
        up = Vec3.up()
        # 3. 2つのベクトルの内積から角度（ラジアン）を求める
        # 浮動小数点誤差で -1.0 〜 1.0 の範囲を超えるのを防ぐ
        dot = max(-1.0, min(1.0, up.dot(normal)))
        angle_rad = math.acos(dot)
        # 回転軸（外積）を求める
        axis = up.cross(normal)

        if axis.length_squared() <= 0.001:
            return None

        axis.normalize()
        quat = Quat()
        quat.set_from_axis_angle_rad(angle_rad, axis)
        return quat.get_hpr()

    def transform_plant_2(self, dummy_np, pos, normal):
        """Using a dummy NodePath, calculate the plant's translation, rotation,
            and scaling, and store the results in a temporary list.
        """
        dummy_np = NodePath('dummy_transform')
        dummy_np.set_pos(pos)
        dummy_np.set_scale(0.1)

        if (hpr := self.calc_hpr(normal)) is None:
            dummy_np.set_h(random.uniform(0, 360))
        else:
            dummy_np.set_hpr(hpr)
            dummy_np.set_h(dummy_np, random.uniform(0, 360))

        mat = dummy_np.get_mat()
        mat_data = [mat.get_cell(r, c) for r in range(4) for c in range(4)]
        return mat_data

    def transform_plant(self, _, pos, normal):
        # 将来的に植物を数万本に増やす場合,さらに高速になる
        quat = Quat()
        if (hpr := self.calc_hpr(normal)) is None:
            quat.set_hpr(Vec3(random.uniform(0, 360), 0, 0))
        else:
            quat.set_hpr(hpr)
            # 法線軸（ローカルZ軸）周りのランダム回転を合成
            spin = Quat()
            spin.set_from_axis_angle(random.uniform(0, 360), Vec3(0, 0, 1))
            quat = spin * quat

        ts = TransformState.make_pos_quat_scale(pos, quat, Vec3(0.1))
        mat = ts.get_mat()
        return [mat.get_cell(r, c) for r in range(4) for c in range(4)]

    def transform_plant_1(self, dummy_np, pos, normal):
        """Using a dummy NodePath, calculate the plant's translation, rotation,
            and scaling, and store the results in a temporary list.
        """
        dummy_np.set_pos(pos)
        dummy_np.set_scale(0.1)
        # dummy_np.set_scale(self.model_scale)

        if normal.length_squared() > 0.001:  # 1e-6
            # If the plant is on a wall, rotate it so that it appears to be growing outward from the wall.
            if (hpr := self.calc_hpr(normal)) is None:
                # dummy_np.set_h(random.uniform(0, 360))
                dummy_np.set_h(dummy_np, random.uniform(0, 360))
                # dummy_np.set_hpr(random.uniform(0, 360), 0, 0)
            else:
                # dummy_np.look_at(pos + normal, Vec3(0, 0, 1))
                dummy_np.set_hpr(hpr)
                # dummy_np.set_h(random.uniform(0, 360))
            # dummy_np.set_p(dummy_np, -90)
                dummy_np.set_h(dummy_np, random.uniform(0, 360))
        else:
            # If the plants are located on the rooftop, place it upright.
            dummy_np.set_hpr(random.uniform(0, 360), 0, 0)

        mat = dummy_np.get_mat()
        # Convert to a flat list.
        mat_data = [mat.get_cell(r, c) for r in range(4) for c in range(4)]
        return mat_data

    def determine_density(self, noise, lower_bound, upper_bound):
        if noise < lower_bound:
            return 0.0

        if noise > upper_bound:
            return 1.0

        # smoothstep
        t = (noise - lower_bound) / (upper_bound - lower_bound)
        density = t * t * (3.0 - 2.0 * t)
        return density




class Scene(NodePath):

    def __init__(self):
        super().__init__(PandaNode('scene'))
        # self.reparent_to(base.render)
        # self.ambient_light = BasicAmbientLight()
        # self.day_light = BasicDayLight()
        # self.sky = Sky()
        # self.sky.reparent_to(self)

        # self.terrain = Terrain('cell_independent_linear_heightmap.png')

        # heightmap_path = 'metora_heightmap.png'
        # heightmap_path = 'terrain/8bit_SimplexNoiseBlending_20261118091850.png'
        # heightmap_path = 'terrain/8bit_SimplexNoiseBlending_20261118092122.png'
        heightmap_path = 'terrain/8bit_SimplexNoiseBlending_20261123102857.png'
        texmap_path = 'terrain/8bit_rock_pillars_texmap_20261109094835.png'
        
        # heightmap_path = 'terrain/8bit_rock_pillars_heightmap_20261109094835.png'
        # texmap_path = 'terrain/8bit_rock_pillars_texmap_20261109094835.png'

        # self.terrain = Terrain(heightmap_path, texmap_path)
        self.terrain = Terrain()

        self.terrain.reparent_to(self)
        self.terrain.set_pos(Point3(0, 0, 0))
        base.world.attach(self.terrain.node())

        self.buildings = NodePath('buildings')
        self.buildings.reparent_to(self)

        # for Building in TownBuilder(texmap_path, self.terrain):
        #     Building.reparent_to(self.buildings)
        #     base.world.attach(Building.node())


        # self.water_surface = WaterSurface()
        # self.water_surface.reparent_to(self)
        # self.water_surface.set_pos(0, 0, 0)
        # base.world.attach(self.water_surface.node())

        # self.road = Road()
        # self.road.set_pos(0, 0, 1)
        # self.road.reparent_to(self)
        # base.world.attach(self.road.node())

    # def update(self, task_time, shadow_target):
    #     self.water_surface.wave(task_time)
    #     self.day_light.update(shadow_target)