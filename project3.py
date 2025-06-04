from OpenGL.GL import *
from glfw.GLFW import *
import glm
import numpy as np



start_x = 0.0
start_y = 0.0
shift = 0.
updown = 0.
azimuth = np.radians(0)
elevation = np.radians(0)
zoom = 0.
obj_path = ""
vao_drop_obj = []
obj_count = float(0)

result = [] #format : name, parent_name, offset, channals
bodyparts = {}
parts_num = 0

Nodes = [] #node list

fname = ""
frames = 0
fps = 0
joint_num = 0
joint_names = []


joint_bodyparts_motions = [] # 모션 한줄 당 Joint transform 행렬 저장

isMouseDown = False
orbitMd, panMd, zoomMd = False, False, False

g_vertex_shader_src = '''
#version 330 core

layout (location = 0) in vec3 vin_pos; 
layout (location = 1) in vec3 vin_normal; 

out vec3 vout_surface_pos;
out vec3 vout_normal;


uniform mat4 MVP;
uniform mat4 M;

void main()
{
    // 3D points in homogeneous coordinates
    vec4 p3D_in_hcoord = vec4(vin_pos.xyz, 1.0);

    gl_Position = MVP * p3D_in_hcoord;

    vout_surface_pos = vec3(M * vec4(vin_pos, 1));
    vout_normal = normalize(mat3(inverse(transpose(M))) * vin_normal);
}
'''


g_fragment_shader_src = '''
#version 330 core

in vec3 vout_surface_pos;
in vec3 vout_normal; //interpolated된 각 fragment의 normal

out vec4 FragColor;

uniform vec3 view_pos;
uniform vec3 mat_color;

void main()
{
    vec3 light_pos = vec3(100, 100, 100);
    vec3 light_color = vec3(0.9, 0.95, 1.0);
    vec3 material_color = mat_color;
    float material_shininess = 16.0;

    vec3 light_ambient = 0.1*light_color;
    vec3 light_diffuse = light_color;
    vec3 light_specular = light_color;

    vec3 material_ambient = material_color;
    vec3 material_diffuse = material_color;
    vec3 material_specular = vec3(1,1,1);

    //ambient
    vec3 ambient = light_ambient * material_ambient;

    //diffiuse and specular
    vec3 normal = normalize(vout_normal);
    vec3 surface_pos = vout_surface_pos;
    vec3 light_dir = normalize(light_pos - surface_pos);

    //diffuse
    float diff = max(dot(normal, light_dir), 0);
    vec3 diffuse = diff * light_diffuse * material_diffuse;

    //specular
    vec3 view_dir = normalize(view_pos - surface_pos);
    vec3 reflect_dir = reflect(-light_dir, normal);
    float spec = pow( max(dot(view_dir, reflect_dir), 0.0), material_shininess);
    vec3 specular = spec * light_specular * material_specular;

    vec3 color = ambient + diffuse + specular;
    FragColor = vec4(color, 1.);
}
'''

def load_shaders(vertex_shader_source, fragment_shader_source):
    # build and compile our shader program
    # ------------------------------------
    
    # vertex shader 
    vertex_shader = glCreateShader(GL_VERTEX_SHADER)    # create an empty shader object
    glShaderSource(vertex_shader, vertex_shader_source) # provide shader source code
    glCompileShader(vertex_shader)                      # compile the shader object
    
    # check for shader compile errors
    success = glGetShaderiv(vertex_shader, GL_COMPILE_STATUS)
    if (not success):
        infoLog = glGetShaderInfoLog(vertex_shader)
        print("ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" + infoLog.decode())
        
    # fragment shader
    fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)    # create an empty shader object
    glShaderSource(fragment_shader, fragment_shader_source) # provide shader source code
    glCompileShader(fragment_shader)                        # compile the shader object
    
    # check for shader compile errors
    success = glGetShaderiv(fragment_shader, GL_COMPILE_STATUS)
    if (not success):
        infoLog = glGetShaderInfoLog(fragment_shader)
        print("ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" + infoLog.decode())

    # link shaders
    shader_program = glCreateProgram()               # create an empty program object
    glAttachShader(shader_program, vertex_shader)    # attach the shader objects to the program object
    glAttachShader(shader_program, fragment_shader)
    glLinkProgram(shader_program)                    # link the program object

    # check for linking errors
    success = glGetProgramiv(shader_program, GL_LINK_STATUS)
    if (not success):
        infoLog = glGetProgramInfoLog(shader_program)
        print("ERROR::SHADER::PROGRAM::LINKING_FAILED\n" + infoLog.decode())
        
    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)

    return shader_program    # return the shader program

def key_callback(window, key, scancode, action, mods):
    global g_cam_ang, g_cam_height
    if key==GLFW_KEY_ESCAPE and action==GLFW_PRESS:
        glfwSetWindowShouldClose(window, GLFW_TRUE);

def mouse_button_callback(window, button, action, mods):
    global isMouseDown, orbitMd, panMd, zoomMd

    if button == GLFW_MOUSE_BUTTON_LEFT:
        if action == GLFW_PRESS:
            if mods & GLFW_MOD_ALT:
                isMouseDown = True
                orbitMd, panMd, zoomMd = False, False, False
                
                if mods & GLFW_MOD_SHIFT: #pan동작
                    panMd = True
                elif mods & GLFW_MOD_CONTROL: #zoom동작
                    zoomMd = True
                else:                       #orbit동작
                    orbitMd = True

        elif action == GLFW_RELEASE:   
            isMouseDown, orbitMd, panMd, zoomMd = False, False, False, False
            

def cursor_position_callback(window, xpos, ypos):
    global start_x, start_y, shift, updown, azimuth, elevation, zoom
    global orbitMd, panMd, zoomMd, isMouseDown
    dx = xpos - start_x
    dy = ypos - start_y
    if isMouseDown:
        if orbitMd:         
            azimuth += np.radians(dx*8)
            elevation -= np.radians(dy*8)
        elif panMd:
            shift += dx*0.1
            updown -= dy*0.1
        elif zoomMd:
            zoom -= dy*0.001
    start_x, start_y = xpos, ypos
    

#--------------------------------------------------------Ear cliping code-------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------------------------#
#vp : vertex positions
#fi : face infomations 
def projection2D(vp, fi): 
    base = glm.vec3(vp[fi[0][0]])
    p1 = glm.vec3(vp[fi[1][0]])
    p2 = glm.vec3(vp[fi[2][0]])

    n = glm.cross(p1 - base, p2 - base)

    u_raw = p1 - base
    if glm.length(u_raw) == 0:
        u = glm.vec3(1.0, 0.0, 0.0)
    else:
        u = glm.normalize(u_raw)

    v = glm.cross(n, u)
    ret_new_p = []
    for i in fi:
        new_p = glm.vec3(vp[i[0]]) - base
        ret_new_p.append((glm.dot(new_p, u), glm.dot(new_p, v)))

    return ret_new_p

def is_convex(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) > 0


def is_in_triangle(p, a, b, c):
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    d1 = sign(p, a, b)
    d2 = sign(p, b, c)
    d3 = sign(p, c, a)

    return (d1 >= 0 and d2 >= 0 and d3 >= 0) or (d1 <= 0 and d2 <= 0 and d3 <= 0)

#i번째 점을 가지는 삼각형이 ear인지 검사
def is_ear(vp_2D, i): 
    if not is_convex(vp_2D[(i-1) % len(vp_2D)], vp_2D[i], vp_2D[(i+1) % len(vp_2D)]):
        return False        #내각이 180도 이상이면 ear가 아님
    
    for j, p in enumerate(vp_2D):
        if j == (i-1) % len(vp_2D) or j == (i+1) % len(vp_2D) or j == i:
            continue
        if is_in_triangle(p, vp_2D[(i-1) % len(vp_2D)], vp_2D[i], vp_2D[(i+1) % len(vp_2D)]):
            return False
    return True

def ear_clipping(vp, fi):
    vp_2D = projection2D(vp, fi)
    ret_indices_normals = []
    while len(vp_2D) > 3:       #다각형이 마지막 삼각형 하나로 쪼개질 떄 까지
        for i in range(len(vp_2D)):
            if is_ear(vp_2D, i):  
                ret_indices_normals.extend([(fi[(i-1) % len(vp_2D)][0], fi[(i-1) % len(vp_2D)][1]),
                                           (fi[i][0], fi[i][1]),
                                           (fi[(i+1) % len(vp_2D)][0], fi[(i+1) % len(vp_2D)][1])]) 
                del fi[i]
                del vp_2D[i]
                break
    ret_indices_normals.extend([(fi[0][0], fi[0][1]),
                               (fi[1][0], fi[1][1]),
                               (fi[2][0], fi[2][1])])
    return ret_indices_normals          #(index, normal) 형식으로 리턴

#--------------------------------------------------------Ear cliping code-------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------------------------#


def drop_list_callback(window, paths):
    global obj_path, vao_drop_obj
    global obj_count, Nodes
    global frames, fps, joint_num, joint_names
    for path in paths:
        obj_path = path
        fm = obj_path.split('.')
        if fm[-1] == 'obj':  # for obj file
            fp = open(obj_path, 'r')
            lines = fp.readlines()

            vertex_positions = []
            vertex_normals = []

            vertex_indices_normals = []

            vertices_3_count = 0
            vertices_4_count = 0
            vertices_4more_count = 0
            face_count = 0
            ccount = 0
            file_name = path.split('/')[-1]
            for line in lines:
                ear_clipping_enable = False
                line_split = line.split()[1:]
                if line.startswith("vn"):
                    vertex_normals.append([float(x) for x in line_split])           
                elif line.startswith("v "):
                    vertex_positions.append([float(x) + obj_count*2 if i == 0 else float(x) for i, x in enumerate(line_split)])
                elif line.startswith("f"):
                    face_count += 1
                    if len(line_split) == 3:
                        vertices_3_count += 1
                    else:
                        if len(line_split) == 4:
                            vertices_4_count += 1
                        elif len(line_split) > 4:
                            vertices_4more_count += 1
                        ear_clipping_enable = True
                        
                    polygon_infomation = []    
                    for x in line_split:
                        parts = x.split('/')
                        v_idx = int(parts[0]) - 1  # vertex index
                        n_idx = int(parts[2]) - 1 if len(parts) >= 3 and parts[2] else -1  # normal index 
                        temp = (v_idx, n_idx)
                        polygon_infomation.append(temp)
                    

                    if ear_clipping_enable:
                        vertex_indices_normals.append(ear_clipping(np.array(vertex_positions), polygon_infomation))
                    else:
                        vertex_indices_normals.append(polygon_infomation)
                print("finish lines !!")
                print("rendering...   ")
                print(ccount/len(lines)*100)
                print("%")
                ccount += 1

            unique_vertex_map = {}  # (v_idx, n_idx) -> new index
            interleaved = []
            ebo = []
            next_index = 0

            for face in vertex_indices_normals:
                for v_idx, n_idx in face:
                    key = (v_idx, n_idx)
                    if key not in unique_vertex_map:
                        unique_vertex_map[key] = next_index
                        next_index += 1

                        pos = vertex_positions[v_idx]  # obj is 1-based
                        norm = vertex_normals[n_idx]

                        interleaved.extend(pos)
                        interleaved.extend(norm)

                    ebo.append(unique_vertex_map[key])

            interleaved = glm.array(glm.float32, *interleaved)
            ebo = glm.array(glm.uint32, *ebo)

            VAO = glGenVertexArrays(1)
            glBindVertexArray(VAO)

            VBO = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, VBO)
            glBufferData(GL_ARRAY_BUFFER, interleaved.nbytes, interleaved.ptr, GL_STATIC_DRAW) 

            EBO = glGenBuffers(1)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, ebo.nbytes, ebo.ptr, GL_STATIC_DRAW)

            #position
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
            glEnableVertexAttribArray(0)

            #normals
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
            glEnableVertexAttribArray(1)
            obj_count += 1
            vao_drop_obj.append([VAO, len(ebo), file_name, face_count, vertices_3_count, vertices_4_count, vertices_4more_count])
            print(f"""            Obj file name                  : {file_name}
                Total number of faces          : {face_count}
                Number of faces with 3 vertices: {vertices_3_count}
                Number of faces with 4 vertices: {vertices_4_count}
                Number of faces with more than 4 vertices: {vertices_4more_count}
                """)
        elif fm[-1] == 'bvh':   #for bvh file
            fp = open(obj_path, 'r')
            lines = fp.readlines()
        
            parent = [None] #stack, first skeletion name is Root(hips)
            value = [] #offset, channels stack
            ch_num = 1

            fname = obj_path.split('/')[-1]
        

            for line in lines:

                sline = line.split()
                if sline[0] == "Frames:":
                    frames = sline[1]
                    print(frames)
                if sline[0] == "Frame" and sline[1] == "Time:":
                    fps = 1/float(sline[2])
                    break

                keyword = sline[0]

                if keyword == "ROOT" or keyword == "JOINT" or keyword == "End":
                    parent.append(sline[1])
                
                if keyword == "OFFSET":
                    value.append(('o', sline[1], sline[2], sline[3]))

                if keyword == "CHANNELS":
                    indices = list(range(ch_num, ch_num + len(sline[2:])))
                    ch_num = ch_num + len(sline[2:])
                    value.append(('c', sline[2:], indices))


                if keyword == '{':
                    value.append(keyword)
                
                if keyword == '}':
                    name = parent.pop()
                    parent_name = parent[-1]
                    channels = ()
                    offset = ()
                    while True:
                        v = value.pop()
                        if v == '{':
                            break
                        else:
                            if v[0] == 'c':
                                channels = v[1:]
                            if v[0] == 'o':
                                offset = v[1:]
                    
                    result.append([name, parent_name, offset, channels])
                    print((name, parent_name, offset, channels))


            registeration()
            print(bodyparts)
            name2index()

            i = 1
            for r in reversed(result):
                if r[0] == None:
                    continue
                link_transform = glm.translate(glm.vec3(float(r[2][0]), float(r[2][1]), float(r[2][2])))
                parent_node = Nodes[r[1]-1] if r[1] else None  # 노드의 부모 설정
                Nodes[parts_num - i] = Node(parent_node,            
                                link_transform, #link transform
                                get_shape_transform_bodypart(r, link_transform),                                 #shape transform
                                glm.vec3(1, 1, 1))                                                       #color 
                i += 1

            joint_num = i-1
            joint_names = list(bodyparts.keys())
            print(f"""            Obj file name                  : {fname}
                Number of frames          : {frames}
                FPS: {fps}
                Number of joints: {joint_num}
                List of all joint names: {joint_names}
                """)
            

            ### setting for MOTION
            motion = 0

            for line in lines:
                sline = line.split()
                if sline[0] == "Frame":
                    motion = 1
                    continue


                if motion:
                    # bodyparts[n] = {조인트 이름 : (joint index, ([zrot xrot yrot], [1, 2, 3]))}
                    xpos, ypos, zpos, xrot, yrot, zrot = 0, 0, 0, 0, 0, 0
                    joint_bodyparts_motion = []
                    for name, (joint_idx, (chinfo, ch)) in bodyparts.items(): 
                        R = glm.mat4()
                        T = glm.mat4()
                        for i, c in enumerate(ch):
                            chname = chinfo[i]
                            chvalue = float(sline[c-1])
                            if chname == "Xposition" or chname == "XPOSITION":
                                xpos = chvalue       # channel c-1가져오기
                            elif chname == "Yposition" or chname == "YPOSITION":
                                ypos = chvalue       
                            elif chname == "Zposition" or chname == "ZPOSITION":
                                zpos = chvalue       
                            elif chname == "Xrotation" or chname == "XROTATION":
                                xrot = chvalue       
                                R = R @ glm.rotate(glm.radians(xrot), (1, 0, 0))
                            elif chname == "Yrotation" or chname == "YROTATION":
                                yrot = chvalue       
                                R = R @ glm.rotate(glm.radians(yrot), (0, 1, 0))
                            elif chname == "Zrotation" or chname == "ZROTATION":
                                zrot = chvalue       
                                R = R @ glm.rotate(glm.radians(zrot), (0, 0, 1))

                        #print(xpos, ypos, zpos)
                        T = glm.translate(glm.vec3(xpos, ypos, zpos))

                        # A Joint matrix / Joint transform  per a bodypart
                        M = T @ R  
                        joint_bodyparts_motion.append(M)

                    joint_bodyparts_motions.append(joint_bodyparts_motion)




                            



            


            




def registeration():
    global bodyparts, parts_num, Nodes
    for p in result:
        if p[0] != "Site":
            parts_num += 1
            bodyparts[p[0]] = (parts_num, p[3])
    Nodes = [None]*parts_num #Node list


def name2index():
    for r in result:
        for i in range(2):
            value = bodyparts.get(r[i])
            if value == None:
                r[i] = None
            else:
                r[i] = bodyparts.get(r[i])[0]
            

# 
# 
def get_shape_transform_bodypart(r, link_transform):
    if r[1] == None:
        return glm.scale(glm.vec3(.01, .01, .01))

    parent_global_position = glm.vec3(0, 0, 0)
    child_global_position = glm.vec3(link_transform * glm.vec4(parent_global_position, 1.0))
    middle_position = (parent_global_position + child_global_position) / 2

    target_vec = glm.normalize(child_global_position - parent_global_position)
    base_vec = glm.vec3(0, 1, 0)

    axis_vec = glm.cross(base_vec, target_vec)
    dot = glm.dot(base_vec, target_vec)

    if glm.length(axis_vec) < 1e-6:
        if dot > 0:
            rotate_mat = glm.mat4()
        else:
            rotate_mat = glm.rotate(glm.pi(), glm.vec3(1, 0, 0))
    else:
        theta = glm.acos(glm.clamp(dot, -1.0, 1.0))
        rotate_mat = glm.rotate(theta, glm.normalize(axis_vec))

    height = glm.length(child_global_position - parent_global_position)
    scale_mat = glm.scale(glm.vec3(.05, height, .05))
    translate_mat = glm.translate(middle_position)

    shape_mat = translate_mat * rotate_mat * scale_mat
    return shape_mat
    

        
def prepare_vao_xyz():
    vertices = [
    #positions        #normals
    glm.array(glm.float32,                  
    -100.0, 0.0, 0.0,  0.0, 1.0, 0.0,  # X-axis
    100.0, 0.0, 0.0,  0.0, 1.0, 0.0,),
    glm.array(glm.float32,
    0.0, -100.0, 0.0,  0.0, 1.0, 0.0,  # Y-axis
    0.0,  100.0, 0.0,  0.0, 1.0, 0.0,),
    glm.array(glm.float32,
    0.0, 0.0, -100.0,  0.0, 1.0, 0.0,  # Z-axis
    0.0, 0.0,  100.0,  0.0, 1.0, 0.0)]

    VAOs = glGenVertexArrays(3)
    VBOs = glGenBuffers(3)

    for i in range(3):
        glBindVertexArray(VAOs[i])
        glBindBuffer(GL_ARRAY_BUFFER, VBOs[i])

        glBufferData(GL_ARRAY_BUFFER, vertices[i].nbytes, vertices[i].ptr, GL_STATIC_DRAW)

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3 * glm.sizeof(glm.float32)))
        glEnableVertexAttribArray(1)

    return VAOs

def prepare_vao_grid():
    vertices = []
    for i in range(201):
        vertices.extend((-100+i , 0, -100, 1, 1, 1)) #(-100,-100) ~ (100, -100)
    for i in range(200):
        vertices.extend((100, 0, -99+i, 1, 1, 1))    
    for i in range(200):
        vertices.extend((99-i, 0, 100, 1, 1, 1))
    for i in range(199):
        vertices.extend((-100, 0, 99-i, 1, 1, 1))
    indices = []
    #가로
    for i in range(100):
        indices.extend((200-i, 400+i)) 
    for i in range(100):
        indices.extend((99-i, 501+i))

    #세로
    for i in range(100):
        indices.extend((400-i, 600+i))
    for i in range(100):
        indices.extend((299-i, (701+i)%800))
    
    vertices_axis = glm.array(glm.float32, *vertices)
    indices_axis = glm.array(glm.uint32, *indices)

    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)

    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertices_axis.nbytes, vertices_axis.ptr, GL_STATIC_DRAW)

    EBO = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices_axis.nbytes, indices_axis.ptr, GL_STATIC_DRAW)

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)  # position
    glEnableVertexAttribArray(0)

    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3 * glm.sizeof(glm.float32)))  #normal
    glEnableVertexAttribArray(1)

    return VAO, len(indices_axis)

def translate(eye, point, tv):
    T = glm.translate(tv)
    return T * eye, T * point

def zoomInOut(eye, wv, zoom):
    Z = glm.translate(glm.mat4(), wv*zoom)
    return Z * eye


#------------------------------------------------------------Node code----------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------------------------#

class Node:
    def __init__(self, parent, link_transform_from_parent, shape_transform, color):
        # hierarchy
        self.parent = parent
        self.children = []
        if parent is not None:
            parent.children.append(self)

        # transform
        self.link_transform_from_parent = link_transform_from_parent
        self.joint_transform = glm.mat4()
        self.global_transform = glm.mat4()

        # shape
        self.shape_transform = shape_transform
        self.color = color

    def set_joint_transform(self, joint_transform):
        self.joint_transform = joint_transform

    def update_tree_global_transform(self):
        if self.parent is not None:
            self.global_transform = self.parent.get_global_transform() * self.link_transform_from_parent * self.joint_transform
        else:
            self.global_transform = self.link_transform_from_parent * self.joint_transform

        for child in self.children:
            child.update_tree_global_transform()

    def get_global_transform(self):
        return self.global_transform
    def get_shape_transform(self):
        return self.shape_transform
    def get_color(self):
        return self.color

def draw_node(vao, node, VP, MVP_loc, color_loc):
    MVP = VP * node.get_global_transform() * node.get_shape_transform()
    color = node.get_color()

    glBindVertexArray(vao)
    
    glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
    glUniform3f(color_loc, color.r, color.g, color.b)
    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, None)


"""
# vertices = glm.array(glm.float32,
        # position      normal
        -1 ,  1 ,  1 , -0.577 ,  0.577,  0.577, # v0
         1 ,  1 ,  1 ,  0.816 ,  0.408,  0.408, # v1
         1 , -1 ,  1 ,  0.408 , -0.408,  0.816, # v2
        -1 , -1 ,  1 , -0.408 , -0.816,  0.408, # v3
        -1 ,  1 , -1 , -0.408 ,  0.408, -0.816, # v4
         1 ,  1 , -1 ,  0.408 ,  0.816, -0.408, # v5
         1 , -1 , -1 ,  0.577 , -0.577, -0.577, # v6
        -1 , -1 , -1 , -0.816 , -0.408, -0.408, # v7
    )
"""
def prepare_vao_box():
   # prepare vertex data (in main memory)
    # 8 vertices
    vertices = glm.array(glm.float32,
        # position      normal
        -1 ,  1 ,  1 , -0.577 ,  0.577,  0.577, # v0
         1 ,  1 ,  1 ,  0.816 ,  0.408,  0.408, # v1
         1 , -1 ,  1 ,  0.408 , -0.408,  0.816, # v2
        -1 , -1 ,  1 , -0.408 , -0.816,  0.408, # v3
        -1 ,  1 , -1 , -0.408 ,  0.408, -0.816, # v4
         1 ,  1 , -1 ,  0.408 ,  0.816, -0.408, # v5
         1 , -1 , -1 ,  0.577 , -0.577, -0.577, # v6
        -1 , -1 , -1 , -0.816 , -0.408, -0.408, # v7
    )

    # prepare index data
    # 12 triangles
    indices = glm.array(glm.uint32,
        0,2,1,
        0,3,2,
        4,5,6,
        4,6,7,
        0,1,5,
        0,5,4,
        3,6,2,
        3,7,6,
        1,2,6,
        1,6,5,
        0,7,3,
        0,4,7,
    )

    # create and activate VAO (vertex array object)
    VAO = glGenVertexArrays(1)  # create a vertex array object ID and store it to VAO variable
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    VBO = glGenBuffers(1)   # create a buffer object ID and store it to VBO variable
    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # activate VBO as a vertex buffer object

    # create and activate EBO (element buffer object)
    EBO = glGenBuffers(1)   # create a buffer object ID and store it to EBO variable
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)  # activate EBO as an element buffer object

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices.ptr, GL_STATIC_DRAW) # allocate GPU memory for and copy vertex data to the currently bound vertex buffer

    # copy index data to EBO
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices.ptr, GL_STATIC_DRAW) # allocate GPU memory for and copy index data to the currently bound element buffer

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex normals
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO



#------------------------------------------------------------Node code----------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------------------------#



def main():
    global shift, updown, azimuth, elevation, zoom
    # initialize glfw
    if not glfwInit():
        return
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3)   # OpenGL 3.3
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3)
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE)  # Do not allow legacy OpenGl API calls
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE) # for macOS

    # create a window and OpenGL context
    window = glfwCreateWindow(800, 800, '2023085605 박수환', None, None)
    
    if not window:
        glfwTerminate()
        return
    glfwMakeContextCurrent(window)

    # register event callbacks
    glfwSetKeyCallback(window, key_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback)
    glfwSetMouseButtonCallback(window, mouse_button_callback)
    glfwSetDropCallback(window, drop_list_callback)

    # load shaders
    shader_program = load_shaders(g_vertex_shader_src, g_fragment_shader_src)

    # get uniform locations
    MVP_loc = glGetUniformLocation(shader_program, 'MVP')
    M_loc = glGetUniformLocation(shader_program, 'M')
    view_pos_loc = glGetUniformLocation(shader_program, 'view_pos')
    mat_color_loc = glGetUniformLocation(shader_program, 'mat_color')
    
    # prepare vaos
    vao_grid = prepare_vao_grid() # return (VAO, len(indices))
    vao_XYZs = prepare_vao_xyz()

    #초기 카메라 세팅
    eye = glm.vec4(20, 20, 20, 1) #직교좌표
    point = glm.vec4(0, 0, 0, 1) #직교좌표
    up = glm.vec4(0,1,0,0)
    azimuth = -135.0    
    elevation = -45.0

    
    vao_box = prepare_vao_box()

    # frame 0 start
    frame = 0 

    # loop until the user closes the window
    while not glfwWindowShouldClose(window):
        # render
        
        # enable depth test (we'll see details later)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)

        glUseProgram(shader_program)

        M = glm.mat4()
        P = glm.perspective(45, 1, 1, 1000)

        # 카메라의 uvw벡터
        cam_w = glm.normalize(glm.vec3(eye - point))
        cam_u = glm.normalize(glm.cross(up.xyz, cam_w))
        cam_v = glm.normalize(glm.cross(cam_w, cam_u))

        #translate
        tv = (cam_v * updown) + (cam_u * shift)
        eye, point = translate(eye, point, tv)
        updown, shift = 0, 0
        #translate


        #rotate
        # 상하(Pitch) 각도 제한
        elevation = max(-89.0, min(89.0, elevation))

        #front 벡터 계산 (yaw = azimuth, pitch = elevation)
        front = glm.vec3(
            glm.cos(glm.radians(azimuth)) * glm.cos(glm.radians(elevation)),
            glm.sin(glm.radians(elevation)),
            glm.sin(glm.radians(azimuth)) * glm.cos(glm.radians(elevation))
        )   #구면좌표계를 사용하여 카메라의 w(front)벡터를 구함. 회전시킨 w벡터를 먼저 구한 후에 camera를 수정
        front = glm.normalize(front)

        #카메라의 좌표계 축 계산(uvw벡터 새로계산)
        right = glm.normalize(glm.cross(front, glm.vec3(0, 1, 0)))  # world up 기준
        up = glm.normalize(glm.cross(right, front))

        #카메라eye 위치 계산
        distance = glm.length(glm.vec3(eye - point)) 
        eye = point - glm.vec4(front * distance, 0)
        #rotate

        #zoom
        # 줌 계산
        Z = glm.translate(glm.vec3(point - eye)*zoom)
        eye = Z * eye
        zoom = 0  # 줌 입력 초기화
        #zoom
        V = glm.lookAt(eye.xyz, point.xyz, up.xyz)

        # current frame: P*V*I (now this is the world frame)
        MVP = P*V*M
        glUniformMatrix4fv(M_loc, 1, GL_FALSE, glm.value_ptr(M))
        glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
        glUniform3f(view_pos_loc, eye.x, eye.y, eye.z)

        # draw current frame
        glBindVertexArray(vao_grid[0])
        glUniform3f(mat_color_loc, 1, 1 ,1)
        glDrawElements(GL_LINES, vao_grid[1], GL_UNSIGNED_INT, None)


        # draw dropped objects
        for vao_obj in vao_drop_obj:    
            glBindVertexArray(vao_obj[0])
            glUniform3f(mat_color_loc, 0.53, 0.81, 0.98) #밝은 하늘색
            glDrawElements(GL_TRIANGLES, vao_obj[1], GL_UNSIGNED_INT, None)
        
        # draw current frame
        color_axis = [glm.vec3(1, 0, 0), glm.vec3(0, 1, 0), glm.vec3(0, 0, 1)]
        for i, vao_axis in enumerate(vao_XYZs):
            c = color_axis[i]
            glUniform3f(mat_color_loc, c.x, c.y, c.z)
            glBindVertexArray(vao_axis)
            glDrawArrays(GL_LINES, 0, 2)


        # draw skeleton

        

        if parts_num > 0: #렌더링할 part(body part)가 하나이상 있으면
            for i in range(parts_num):
                
                    # 0 : translate   1: rotate(euler)
                #print(joint_bodyparts_motions[frame][i])
                Nodes[i].set_joint_transform(joint_bodyparts_motions[frame][i])

            Nodes[parts_num-1].update_tree_global_transform()
            for i in range(parts_num):
                #print(Nodes[i].get_global_transform())
                glUniformMatrix4fv(M_loc, 1, GL_FALSE, glm.value_ptr(Nodes[i].get_shape_transform()))
                draw_node(vao_box, Nodes[i], P*V, MVP_loc, mat_color_loc)
            
            # 만약 프레임이 199개이면, 0 <= frame <= 198 
            frame = (frame + 1)%int(frames)

        # swap front and back buffers
        glfwSwapBuffers(window)

        # poll events
        glfwPollEvents()


    # terminate glfw
    glfwTerminate()

if __name__ == "__main__":
    main()