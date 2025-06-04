import os

input_path = "mesh.obj"  # 입력 파일 경로
output_dir = "split_objs"
os.makedirs(output_dir, exist_ok=True)

with open(input_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()

header = []
objects = {}
current_obj_name = None
current_obj_data = []
last_usemtl = None

for line in lines:
    if line.startswith('mtllib'):
        header.append(line)
    elif line.startswith('o '):
        # 이전 오브젝트 저장
        if current_obj_name:
            objects[current_obj_name] = header + current_obj_data
            current_obj_data = []
            last_usemtl = None  # reset material tracking

        # 새로운 오브젝트 시작
        current_obj_name = line.strip().split(' ', 1)[1]
        current_obj_data.append(line)
    elif line.startswith('usemtl'):
        material = line.strip()
        if material != last_usemtl:
            current_obj_data.append(line)
            last_usemtl = material
        # else: 중복이므로 저장하지 않음
    else:
        current_obj_data.append(line)

# 마지막 오브젝트 저장
if current_obj_name:
    objects[current_obj_name] = header + current_obj_data

# 각각 저장
for name, data in objects.items():
    safe_name = name.replace(":", "_")
    with open(os.path.join(output_dir, f"{safe_name}.obj"), 'w', encoding='utf-8') as f:
        f.writelines(data)

print(f"✅ {len(objects)}개의 OBJ 파일이 중복 없는 상태로 저장되었습니다.")
