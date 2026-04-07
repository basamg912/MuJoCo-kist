#include <stdio.h>
#include <mujoco/mujoco.h>
#include <mujoco/mjplugin.h>

int main(int argc, char** argv) {
    if (argc != 3) {
        printf("Usage: urdf2mjcf input.urdf output.xml\n");
        return 1;
    }

    // STL 디코더 등 플러그인 로드
    mj_loadAllPluginLibraries("/home/kist/mujoco-3.6.0/bin/mujoco_plugin", NULL);

    char error[1000] = {0};

    // URDF 로드
    mjModel* m = mj_loadXML(argv[1], NULL, error, 1000);
    if (!m) {
        printf("Load error: %s\n", error);
        return 1;
    }

    // MJCF XML로 저장
    if (!mj_saveLastXML(argv[2], m, error, 1000)) {
        printf("Save error: %s\n", error);
        mj_deleteModel(m);
        return 1;
    }

    printf("Converted: %s -> %s\n", argv[1], argv[2]);
    mj_deleteModel(m);
    return 0;
}
