import math
import random
from enum import StrEnum

import cv2
import numpy as np

from panda3d.bullet import BulletRigidBodyNode
from panda3d.bullet import BulletConvexHullShape, BulletHeightfieldShape, ZUp
from panda3d.core import NodePath, PandaNode
from panda3d.core import BitMask32, Vec3, Point3, LColor
from panda3d.core import PNMImage, Texture, TextureStage
from panda3d.core import SamplerState, TransformState
from panda3d.core import GeoMipTerrain
from panda3d.core import Shader, ShaderBuffer, GeomEnums
from panda3d.core import OmniBoundingVolume, Quat

from texture_generator.heightmap_generator import RoundedVoronoiRockPillarsHeightmap
from shapes import RandomPolygonalPrism, Sphere
from texture_generator.noise_compositing import SimplexNoiseBlending
from texture_generator.heightmap_generator import ProcessHeightmapMixin
from voronoi_generator.voronoi_2d import Polygon2DMixin
from voronoi_generator.voronoi_2d import VoronoiSitesGenerator, RoundedVoronoiGenerator


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


class Terrain:

    def __init__(self, parent, height=100):
        self.parent = parent
        self.terrain_np = NodePath(PandaNode('terrain'))
        self.height = height

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

    def create_heightmap_and_texmap(self, size, heightmap_path, texmap_path, debug):
        heightmap_img = texmap_img = None

        if heightmap_path and texmap_path:
            if (heightmap_img := cv2.imread(heightmap_path)) is None:
                print(f'Heightmap not found or could not be read: {heightmap_path}')

            if (texmap_img := cv2.imread(texmap_path)) is None:
                print(f'Texmap not found or could not be read: {texmap_path}')

        if heightmap_img is None or texmap_img is None:
            t = random.uniform(0, 1000)

            maker = RoundedVoronoiRockPillarsHeightmap(grid=3, size=size, blur=None, tp=10, t=t, random_height=True)
            arr = maker.create_heightmap(bit=8)
            blender = SimplexNoiseBlending(debug=False, noise_scale=100, noise_intensity=100)
            arr = blender.blend(arr)
            heightmap_img = np.stack([arr] * 3, axis=-1)

            maker = RoundedVoronoiRockPillarsHeightmap(grid=3, size=size, blur=None, tp=10, t=t, random_height=False)
            arr = maker.create_heightmap(bit=8)
            texmap_img = np.stack([arr] * 3, axis=-1)

            if debug:
                cv2.imwrite('heightmap.png', heightmap_img)
                cv2.imwrite('texmap.png', texmap_img)

        return heightmap_img, texmap_img

    def generate(self, heightmap_path=None, texmap_path=None, size=257, debug=False):
        heightmap_img, texmap_img = self.create_heightmap_and_texmap(
            size, heightmap_path, texmap_path, debug)

        self.create_terrain(heightmap_img)
        self.setup_shader(texmap_img)
        return heightmap_img, texmap_img

    def create_terrain(self, img):
        img = cv2.GaussianBlur(img, (11, 11), 0)
        tex_heightfield = self.generate_texture_from_img(img)
        # Convert the texture to PNMImage.
        pnm_img = PNMImage()
        tex_heightfield.store(pnm_img)

        self.terrain = GeoMipTerrain('geomip_terrain')
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
        self.root.reparent_to(self.terrain_np)

        pos = Point3(0)
        hpr = Vec3(00)
        shape = BulletHeightfieldShape(pnm_img, self.height, ZUp)
        shape.set_use_diamond_subdivision(True)
        self.parent.node().add_shape(shape, TransformState.make_pos_hpr(pos, hpr))
        self.terrain_np.set_pos_hpr(pos, hpr)
        self.terrain_np.reparent_to(self.parent)

    def setup_shader(self, img):
        img = cv2.GaussianBlur(img, (11, 11), 0)
        texmap = self.generate_texture_from_img(img)
        # Convert the texture to PNMImage.
        shader = Shader.load(Shader.SL_GLSL, 'shaders/terrain_v.glsl', 'shaders/terrain_f.glsl')
        self.root.set_shader(shader)

        self.root.set_shader_input("height_scale", self.height)
        self.root.set_shader_input("terrain_size", 256)
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


class Building(Polygon2DMixin):

    def __init__(self, parent, foundation_h, wall_h, belt_h):
        self.parent = parent
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
        self.parent.node().add_shape(shape, TransformState.make_pos(pos))

        model.set_pos(pos)
        model.set_name(name)
        model.reparent_to(self.parent)

    def get_terrain_z(self, world_cx, world_cy):
        from_pos = Point3(world_cx, world_cy, 150)
        to_pos = Point3(world_cx, world_cy, 0)

        if (result := base.world.ray_test_closest(
                from_pos, to_pos, BitMask32.bit(1))).has_hit():
            return result.get_hit_pos()

    def build(self, building_id, vertices, textures, offset):
        # Determine the value of segs_top_cap based on the vertex farthest from the center.
        _, max_distance = self.get_max_distance_from_center(vertices)
        segs_cap = 3 if max_distance <= 2 else int(max_distance / 2)

        wall_creator = RandomPolygonalPrism(list(vertices), segs_top_cap=0, segs_bottom_cap=0)
        belt_creator = RandomPolygonalPrism(list(vertices * 1.02), height=self.belt_h, segs_top_cap=segs_cap, segs_bottom_cap=segs_cap)

        foundation_pos = Point3(*wall_creator.center) + offset
        hit_pos = self.get_terrain_z(foundation_pos.x, foundation_pos.y)
        foundation_pos.z = hit_pos.z - 10
        self.create_foundation(building_id, foundation_pos, wall_creator, textures)

        wall_pos = foundation_pos + Vec3(0, 0, self.foundation_h)
        self.create_wall(building_id, wall_pos, wall_creator, belt_creator, textures)

    def create_foundation(self, building_id, pos, wall_creator, textures):
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

        self.assemble(model, pos, name=f"foundtion_{building_id}")

    def create_wall(self, building_id, pos, wall_creator, belt_creator, textures):
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
            self.assemble(wall, pos, f'wall{i}_{building_id}')
            pos.z += wall_creator.height

            # Create belt between walls.
            belt = belt_creator.create()
            belt.set_texture(tex_belt)
            self.assemble(belt, pos, f'belt{i}_{building_id}')
            pos.z += belt_creator.height


class Buildings(ProcessHeightmapMixin, Building):

    def __init__(self, parent, area_ratio=0.75, foundation_h=15, wall_h=10, belt_h=2):
        super().__init__(parent, foundation_h, wall_h, belt_h)
        self.area_ratio = area_ratio
        self.create_textures()

    def load_tex(self, file_name):
        return base.loader.load_texture(f'textures/{file_name}')

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

    def create_building(self, num, img_size, sorted_pts, offset_pos):
        stories = random.randint(1, 3)
        self.tex_building['walls'] = self.tex_walls[:1] * stories if stories <= 2 \
            else [self.tex_walls[1]] * (stories - 1) + [self.tex_walls[2]]

        scaled_pts = sorted_pts * img_size * self.area_ratio
        self.build(num, scaled_pts, self.tex_building, offset_pos)

    def generate(self, heightmap_img):
        if heightmap_img.ndim == 3:
            img = heightmap_img[:, :, 0]

        img = cv2.flip(img, 0)
        h, w = img.shape[:2]
        offset_cx, offset_cy = w / 2.0, h / 2.0

        non_builing_contours = []

        for i, (center, vertices) in enumerate(self.generate_contour_vertices(img)):
            # ###########################
            if i % 2 == 1:
                non_builing_contours.append((center, vertices))
                continue
            # ###########################
            world_cx = center[0] - offset_cx
            world_cy = offset_cy - center[1]

            local_vertices = vertices - center
            local_vertices[:, 1] = -local_vertices[:, 1]

            region = (local_vertices / h) * self.area_ratio
            sites = np.array([pt for pt in VoronoiSitesGenerator(region)])
            offset_pos = Point3(world_cx, world_cy, 0)

            # Generate vertex coordinates of a rounded-corner Voronoi cell.
            for i, pts in enumerate(RoundedVoronoiGenerator(
                    pts=sites, bnd=region, segment_length=0.0029)):

                if len(pts) == 0:
                    continue

                polygon = np.insert(pts, pts.shape[1], 0, axis=1)
                sorted_pts = self.sort_counter_clockwise(polygon)
                self.create_building(i, h, sorted_pts, offset_pos)

        return non_builing_contours


class Vegetation:

    def __init__(self, parent, area_ratio=0.6, vegetation_threshold=0.5, num_samples=300): 
        self.parent = parent
        self.area_ratio = area_ratio
        self.vegetation_threshold = vegetation_threshold
        self.num_samples = num_samples

    def create_vegetation_map(self, heightmap_arr, contours_li):
        """Create a vegetation map from hegitmap image.
            Args:
                heightmap_arr (numpy.ndarray): A grayscale heightmap image.
                contours_li (list): A list of (center, contour vertices).
                    center is a numpy.ndarray whose shape is (2,).
                    contour vertices is a numpy.ndarray that stores xy coordinates.
        """
        contours = [((verts - center) * self.area_ratio + center).astype(np.int32)
                    for center, verts in contours_li]

        heightmap_img = heightmap_arr[:, :, 0]
        h, w = heightmap_img.shape[:2]

        contours_img = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(contours_img, contours, -1, color=255, thickness=cv2.FILLED)
        contours_img = cv2.flip(contours_img, 0)

        vegetation_map = np.zeros((h, w), dtype=np.uint8)
        mask = contours_img == 255
        vegetation_map[mask] = heightmap_img[mask]

        v_min = vegetation_map.min()
        v_max = vegetation_map.max()
        vegetation_map = (vegetation_map - v_min) / (v_max - v_min)

        return vegetation_map

    def generate(self, heightmap_arr, contours_li):
        vegetation_map = self.create_vegetation_map(heightmap_arr, contours_li)
        # cv2.imwrite('vegetation_map2.png', vegetation_map*255)
        coords = np.argwhere(vegetation_map >= self.vegetation_threshold)

        if len(coords) > 0:
            h, w = heightmap_arr.shape[:2]
            offset_x = w / 2
            offset_y = h / 2

            sample_size = min(len(coords), self.num_samples)
            indices = np.random.choice(len(coords), size=sample_size, replace=False)
            selected_coords = coords[indices]
            mat_plants = []

            for y, x in selected_coords:
                normal = self.parent.terrain.calc_normal_from_heightmap(x, y)
                z = self.parent.terrain.get_terrain_z(x, y)
                pos = Point3(x - offset_x, y - offset_y, z)

                mat_data = self.transform_plant(pos, normal)
                mat_plants.append(mat_data)

            self.create_model(mat_plants)

    def create_model(self, matrices, color_scale=None):
        arr_matrices = np.array(matrices, dtype=np.float32)
        raw_buffer_data = arr_matrices.tobytes()
        model = base.loader.load_model('models/plants1/plants1.egg')
        # model = base.loader.load_model('models/shrubbery/shrubbery.egg')

        if color_scale is not None:
            # Set the color to white first, since the color of plant1 didn't change with just set_color_scale.
            model.set_color(1, 1, 1, 1)
            model.set_color_scale(*color_scale, 1)

        model.flatten_light()
        model.reparent_to(self.parent)
        model.set_pos_hpr(Point3(0), Vec3(0))

        # Prevent curling.
        for geom_np in model.find_all_matches("**/+GeomNode"):
            geom_nd = geom_np.node()
            geom_nd.set_bounds(OmniBoundingVolume())
            geom_nd.set_final(True)

        # Set the number of instances.
        instance_cnt = len(matrices)
        model.set_instance_count(instance_cnt)
        # Set shader.
        model.set_shader(Shader.load(Shader.SL_GLSL, vertex='shaders/instancing_v.glsl', fragment='shaders/instancing_f.glsl'))
        model.set_shader_input("instanced_object", ShaderBuffer('DataBuffer', raw_buffer_data, GeomEnums.UH_static))

    def calc_hpr(self, normal):
        up = Vec3.up()
        # Calculate the angle in radians from the dot product of two vectors,
        # and prevent the result from exceeding the range of -1.0 to 1.0 due to floating-point error.
        dot = max(-1.0, min(1.0, up.dot(normal)))
        angle_rad = math.acos(dot)
        # Find the rotation axis (cross product).
        axis = up.cross(normal)

        if axis.length_squared() <= 0.001:
            return None

        axis.normalize()
        quat = Quat()
        quat.set_from_axis_angle_rad(angle_rad, axis)
        return quat.get_hpr()

    def transform_plant(self, pos, normal):
        """Using a dummy NodePath, calculate the plant's translation, rotation, and scaling.
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

    def transform_plant_fast(self, _, pos, normal):
        """Runs quickly because it does not use the dummy NodePath.
        """
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


class SkyRockPillars(NodePath):

    def __init__(self):
        super().__init__(BulletRigidBodyNode('rock_pillars'))
        self.node().set_mass(0)
        self.set_collide_mask(BitMask32.bit(1))

    def generate(self, size=257, height=100, heightmap_path=None, texmap_path=None, debug=False):
        self.terrain = Terrain(parent=self, height=height)
        heightmap_img, texmap_img = self.terrain.generate(heightmap_path, texmap_path, size, debug)
        base.world.attach(self.node())

        buildings = Buildings(parent=self)
        non_builing_contours = buildings.generate(texmap_img)

        vegetation = Vegetation(self)
        vegetation.generate(heightmap_img, non_builing_contours)


class Scene(NodePath):

    def __init__(self):
        super().__init__(PandaNode('scene'))
        # self.reparent_to(base.render)
        # self.ambient_light = BasicAmbientLight()
        # self.day_light = BasicDayLight()
        # self.sky = Sky()
        # self.sky.reparent_to(self)

        heightmap_path = 'terrain/8bit_SimplexNoiseBlending_20261123102857.png'
        texmap_path = 'terrain/8bit_rock_pillars_texmap_20261109094835.png'
        rock_pillars = SkyRockPillars()
        rock_pillars.generate(heightmap_path=heightmap_path, texmap_path=texmap_path)

        rock_pillars.reparent_to(base.render)
        rock_pillars.set_pos(0, 0, rock_pillars.terrain.height / 2)

        sphere = Sphere(radius=20).create()
        sphere.set_color(1, 0, 0, 1)
        sphere.reparent_to(base.render)
        sphere.set_pos(Point3(0, -128, 0))

        # size = 257
        # half_size = (size - 1) / 2

        # for x, y in [(half_size, half_size), (half_size, -half_size), (-half_size, half_size), (-half_size, -half_size)]:
        #     rock_pillars = SkyRockPillars()
        #     rock_pillars.generate(size=size, height=100)
        #     rock_pillars.reparent_to(base.render)
        #     rock_pillars.set_pos(x, y, rock_pillars.terrain.height / 2)

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