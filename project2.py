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

vertex_positions_array = []
vertex_normals_array = []
vertex_face_normals_array = []
vertex_indices_array = []

isMouseDown = False
orbitMd, panMd, zoomMd = False, False, False

g_vertex_shader_src = '''
#version 330 core

layout (location = 0) in vec3 vin_pos; 
layout (location = 1) in vec3 vin_color; 

out vec4 vout_color;

uniform mat4 MVP;

void main()
{
    // 3D points in homogeneous coordinates
    vec4 p3D_in_hcoord = vec4(vin_pos.xyz, 1.0);

    gl_Position = MVP * p3D_in_hcoord;

    vout_color = vec4(vin_color, 1.);
}
'''

g_fragment_shader_src = '''
#version 330 core

in vec4 vout_color;

out vec4 FragColor;

void main()
{
    FragColor = vout_color;
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
    
#vp : vertex positions
#fi : face infomations 
def projection2D(vp, fi): 
    base = glm.vec3(vp[fi[0][0]])
    p1 = glm.vec3(vp[fi[1][0]])
    p2 = glm.vec3(vp[fi[2][0]])

    n = glm.cross(p1 - base, p2 - base)

    # u: 기준 축 벡터 (p1 - base) 정규화
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
    print(vp_2D)
    indices = []        #삼각분할 한 후의 indices
    normals = []        #위 indices에 맞는 normal
    while len(vp_2D) > 3:       #다각형이 삼각형으로 쪼개질 떄 까지
        for i in range(len(vp_2D)):
            if is_ear(vp_2D, i):        
                indices.append(fi[i][0])
                normals.append(fi[i][1])
                del fi[i]
                del vp_2D[i]
                break
    indices.extend([fi[0][0], fi[1][0], fi[2][0]])
    normals.extend([fi[0][1], fi[1][1], fi[2][1]])        #마지막 남은 삼각형을 저장
    return indices, normals     #삼각분할한 인덱스들만을 반환(vertex position은 고정)

def drop_list_callback(window, paths):
    global obj_path, vao_drop_obj
    global obj_count
    for path in paths:
        obj_path = path
        fp = open(obj_path, 'r')
        lines = fp.readlines()

        vertex_positions = []
        vertex_normals = []
        vertex_indices = []
        vertex_face_normals = []

        vertices_3_count = 0
        vertices_4_count = 0
        vertices_4more_count = 0
        face_count = 0
        file_name = path.split('/')[-1]
        for line in lines:
            ear_clipping_enable = False
            line_split = line.split()[1:]
            if line.startswith("vn"):
                vertex_normals.extend([float(x) for x in line_split])           
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
                    polygon_vertex_indices, polygon_face_infomations = ear_clipping(np.array(vertex_positions), polygon_infomation)
                    vertex_indices.extend(polygon_vertex_indices)
                    vertex_face_normals.extend(polygon_face_infomations)
                    #print(vertex_indices)
                else:
                    vertex_indices.extend([v for v, _ in polygon_infomation])
                    vertex_face_normals.extend([n for _, n in polygon_infomation])


        
        #positions는 전처리를 위해 2차원 배열로 저장되어 있었음
        flat = [coord for vertex in vertex_positions for coord in vertex]
        vertex_positions = glm.array(glm.float32, *flat) 
        vertex_normals = glm.array(glm.float32, *vertex_normals)
        vertex_indices = glm.array(glm.uint32, *vertex_indices)
        vertex_face_normals = glm.array(glm.int32, *vertex_face_normals)

        VAO = glGenVertexArrays(1)
        glBindVertexArray(VAO)

        VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, VBO)
        glBufferData(GL_ARRAY_BUFFER, vertex_positions.nbytes, vertex_positions.ptr, GL_STATIC_DRAW) 

        EBO = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, vertex_indices.nbytes, vertex_indices.ptr, GL_STATIC_DRAW)

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * glm.sizeof(glm.float32), None)
        glEnableVertexAttribArray(0)
        obj_count += 1
        vao_drop_obj.append([VAO, len(vertex_indices), file_name, face_count, vertices_3_count, vertices_4_count, vertices_4more_count])
        print(f"""            Obj file name                  : {file_name}
            Total number of faces          : {face_count}
            Number of faces with 3 vertices: {vertices_3_count}
            Number of faces with 4 vertices: {vertices_4_count}
            Number of faces with more than 4 vertices: {vertices_4more_count}
            """)
    
        

def prepare_vao_rectangle():
    v = [
        (0.5, -0.5, 0.5),
        (-0.5, -0.5, 0.5),
        (-0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5),
        (0.5, -0.5, -0.5),
        (-0.5, -0.5, -0.5),
        (-0.5, 0.5, -0.5),
        (0.5, 0.5, -0.5)
    ]
    c = [.5, .5, .5]
    # prepare vertex data (in main memory)
    vertices = glm.array(glm.float32,
        *v[0], *c, *v[2], *c, *v[1], *c, #triangle 1
        *v[0], *c, *v[3], *c, *v[2], *c,
        *v[0], *c, *v[4], *c, *v[3], *c,
        *v[3], *c, *v[4], *c, *v[7], *c,
        *v[3], *c, *v[7], *c, *v[6], *c,
        *v[3], *c, *v[6], *c, *v[2], *c,
        *v[2], *c, *v[6], *c, *v[5], *c,
        *v[2], *c, *v[5], *c, *v[1], *c,
        *v[1], *c, *v[5], *c, *v[0], *c,
        *v[0], *c, *v[5], *c, *v[4], *c,
        *v[5], *c, *v[6], *c, *v[7], *c,  
        *v[5], *c, *v[7], *c, *v[4], *c   #triangle 12
    )

    # create and activate VAO (vertex array object)
    VAO = glGenVertexArrays(1)  # create a vertex array object ID and store it to VAO variable
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    VBO = glGenBuffers(1)   # create a buffer object ID and store it to VBO variable
    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # activate VBO as a vertex buffer object

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices.ptr, GL_STATIC_DRAW) # allocate GPU memory for and copy vertex data to the currently bound vertex buffer

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex colors
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO

def prepare_vao_frame():
    # prepare vertex data (in main memory)
    vertices = [
        0.0, 0.0, 0.0,  1.0, 0.0, 0.0, # x-axis start
         50.0, 0.0, 0.0,  1.0, 0.0, 0.0, # x-axis end 
         0.0, 0.0, 0.0,  0.0, 1.0, 0.0, # y-axis start
         0.0, 50.0, 0.0,  0.0, 1.0, 0.0, # y-axis end 
         0.0, 0.0, 0.0,  0.0, 0.0, 1.0, # z-axis start
         0.0, 0.0, 50.0,  0.0, 0.0, 1.0] # z-axis end
    
    x = -100.0
    z = 100.0
    while x<=100.0:
        vertices.extend([round(x, 2), 0.0, z, 0.5, 0.5, 0.5])
        vertices.extend([round(x, 2), 0.0, -z, 0.5, 0.5, 0.5])
        x += 1
    x = 100.0
    z = -100.0
    while z<=100.0:
        vertices.extend([x, 0.0, round(z, 2), 0.5, 0.5, 0.5])
        vertices.extend([-x, 0.0, round(z, 2), 0.5, 0.5, 0.5])
        z += 1


    vertices_axis = glm.array(glm.float32, *vertices)
    # create and activate VAO (vertex array object)
    VAO = glGenVertexArrays(1)  # create a vertex array object ID and store it to VAO variable
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    VBO = glGenBuffers(1)   # create a buffer object ID and store it to VBO variable
    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # activate VBO as a vertex buffer object

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, vertices_axis.nbytes, vertices_axis.ptr, GL_STATIC_DRAW) # allocate GPU memory for and copy vertex data to the currently bound vertex buffer

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex colors
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO

def translate(eye, point, tv):
    T = glm.translate(tv)
    return T * eye, T * point

def zoomInOut(eye, wv, zoom):
    Z = glm.translate(glm.mat4(), wv*zoom)
    return Z * eye


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
    
    # prepare vaos
    vao_rectangle = prepare_vao_rectangle()
    vao_frame = prepare_vao_frame()
    
    #초기 카메라 세팅
    eye = glm.vec4(20, 20, 20, 1) #직교좌표
    point = glm.vec4(0, 0, 0, 1) #직교좌표
    up = glm.vec4(0,1,0,0)
    azimuth = -135.0    
    elevation = -45.0


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
        glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
        # draw current frame
        glBindVertexArray(vao_frame)
        glDrawArrays(GL_LINES, 0, 1000)

        for vao_obj in vao_drop_obj:
            glBindVertexArray(vao_obj[0])
            glDrawElements(GL_TRIANGLES, vao_obj[1], GL_UNSIGNED_INT, None)
        

        # draw triangle w.r.t. the current frame
        #glBindVertexArray(vao_rectangle)
        #glDrawArrays(GL_TRIANGLES, 0, 3*12)

        # draw current frame
        glBindVertexArray(vao_frame)
        glDrawArrays(GL_LINES, 0, 6)

        # swap front and back buffers
        glfwSwapBuffers(window)

        # poll events
        glfwPollEvents()

    # terminate glfw
    glfwTerminate()

if __name__ == "__main__":
    main()