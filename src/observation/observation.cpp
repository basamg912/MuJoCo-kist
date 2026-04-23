#include "observation.h"
#include <iostream>
#include "mapping.h"



Observation::Observation()
    : _mj_model(nullptr), _mj_data(nullptr), _pelvis_body_id(-1)
{
    _vel_cmd.setZero();
    // ? Eigen::Vector3d method overload
    _default_com << -0.066071861, -0.19773669, 0.0024721903;
}

Observation::~Observation(){}

void Observation::setMujocoModel(const mjModel* m, mjData* d){
    _mj_model = m;
    _mj_data = d;

    _pelvis_body_id = mj_name2id(m, mjOBJ_BODY, "pelvis");
    if(_pelvis_body_id <0)  std::cout << "[INFO] Pelvis body not found" <<'\n';
    std::cout << "[INFO] Body num : " << m->nbody << '\n';
    std::cout<<"[INFO] Pelvis ID : " << _pelvis_body_id << '\n';
    for (int i=0; i< m->nbody; i++){
        const char* name = mj_id2name(m, mjOBJ_BODY, i);
        std::cout << "[INFO] body id "<<"'" << i<< "'" << " name : " << name <<'\n';
    }
}

void Observation::setComCommand(double dx, double dy, double dz){
    _com_cmd << dx, dy, dz;
}
void Observation::setVelocityCommand(double vx, double vy, double wz){
    _vel_cmd << vx, vy, wz;
}

void Observation::reset(){
    _history.clear(); // ? deque clear
}


Eigen::VectorXd Observation::computeSingleObs(
    const Eigen::VectorXd& q,
    const Eigen::VectorXd& qdot,
    const Eigen::VectorXd& q_home,
    const Eigen::VectorXd& last_action
)
{
    Eigen::VectorXd obs(SINGLE_DIM);
    int idx=0;

    // * 1) ang vel
    // ? Eigen::Vector 는 벡터, mjtNum 은 array(raw array)
    // ? vector 는 행렬연산 용, array 는 값 저장용
    mjtNum ang_vel_world[3] = {
        _mj_data->qvel[3], _mj_data->qvel[4], _mj_data->qvel[5]
    };
    
    // ? quat 는 자세를 결정, 모든 바디에 대한 회전 정보를 저장
    // ! pelvis_body_id 가 순서에 맞는지 확인 필요
    mjtNum base_quat[4] = {
        _mj_data->xquat[4*_pelvis_body_id +0],
        _mj_data->xquat[4*_pelvis_body_id +1],
        _mj_data->xquat[4*_pelvis_body_id +2],
        _mj_data->xquat[4*_pelvis_body_id +3],
    };
    mjtNum base_quat_inv[4];
    // ? quat 부호 반전, conjugate 를 계산 -> 정반대 방향
    mju_negQuat(base_quat_inv, base_quat);
    
    // ? ang_vel_world 를 쿼터니언 만큼 회전시킨 것을 ang_vel_body 에 저장
    // ? world 기준 값 -> body 기준 로컬 값 변환
    mjtNum ang_vel_body[3];
    mju_rotVecQuat(ang_vel_body, ang_vel_world, base_quat_inv);
    for (int i=0; i<3; i++) obs(idx++) = ang_vel_body[i]*0.2; // ! 0.2 는 스케일링 용도 -> 신경망 input
    
    // * 2) gravity
    // ? 월드좌표상으로는 -z 방향 -> 로봇의 기울어진 방향 기준으로 변경. 중력의 힘보다 방향을 중요시 -> norm 을 1로
    mjtNum gravity_world[3] = {0.0, 0.0, -1.0};
    mjtNum gravity_body[3];
    mju_rotVecQuat(gravity_body, gravity_world, base_quat_inv);
    for(int i=0; i<3; i++)  obs(idx++) = gravity_body[i];

    // * 3) vel
    for (int i=0; i<3; i++) obs[idx++] = _vel_cmd[i];

    // ! 14개 다리에 대해서만 policy action 적용, 나머지는 포지션 제어
    // // * 4) joint_pos_rel
    // for (int i=0; i<LEG_DIM; i++)   obs[idx++]= q(isaac_leg_to_mujoco[i]) - q_home(isaac_leg_to_mujoco[i]);

    // // * 5) joint_pos_rel
    // for (int i=0; i<LEG_DIM; i++)   obs[idx++] = qdot(isaac_leg_to_mujoco[i]) * 0.05;

    // ! 4/23 - 모든 조인트가 observation 으로 들어가도록
    // * 4) joint_pos_rel
    for (int i=0; i<JOINT_DIM; i++)   obs[idx++]= q(isaac_joint_to_mujoco[i]) - q_home(isaac_joint_to_mujoco[i]);

    // * 5) joint_pos_rel
    for (int i=0; i<JOINT_DIM; i++)   obs[idx++] = qdot(isaac_joint_to_mujoco[i]) * 0.05;

    // * 6) last_action
    for (int i=0; i<last_action.size(); i++)    obs[idx++] = last_action(i);

    // * 7) torso_com dif
    // ! 수정 필요 -> WL3 에 대해서만 적용되게
    // ! body_id 0 은 world body 를 의미, 1부터 pelvis
    int wl3_id = mj_name2id(_mj_model, mjOBJ_BODY, "WL3");
    for(int i=0; i<3; i++){
        obs(idx++) = _com_cmd[i];
    }
        // ! debug 용, first observation
    // static bool once = true;
    // if (once) {
    //     std::cout << "ang_vel: " << obs.segment(0, 3).transpose() << std::endl;
    //     std::cout << "gravity: " << obs.segment(3, 3).transpose() << std::endl;
    //     std::cout << "joint_pos: " << obs.segment(9, 14).transpose() << std::endl;
    //     std::cout << "torso_com: " << obs.segment(68, 3).transpose() << std::endl;
    //     once = false;
    // }

    return obs;
}

Eigen::VectorXd Observation::update(
    const Eigen::VectorXd& q,
    const Eigen::VectorXd& qdot,
    const Eigen::VectorXd& q_home,
    const Eigen::VectorXd& last_action
)
{
    Eigen::VectorXd single_obs = computeSingleObs(q, qdot, q_home, last_action);

    if(_history.empty()){
        for (int i=0; i<HISTORY_LEN; i++){
            _history.push_back(single_obs);
        }
    }
    else{
        _history.push_back(single_obs);
        if ( (int)_history.size() > HISTORY_LEN ) _history.pop_front();
    }

    Eigen::VectorXd stacked(STACKED_DIM);
    int idx=0;
    // ! history stack
    // joint_pos_rel / joint_vel_rel 를 31로 확장한 버전
    int term_starts[] = {0, 3, 6, 9, 40, 71, 102};
    int term_size[]   = {3, 3, 3, 31, 31, 31, 3};
    int num_terms = 7;

    // for (int i=0; i<num_terms; i++){
    //     for (int j=0; j<HISTORY_LEN; j++){
    //         stacked.segment(idx, term_size[i]) = _history[j].segment(term_starts[i], term_size[i]);
    //         idx += term_size[i];
    //     }
    // }
    // ! 역순 history
    for (int i=0; i<num_terms; i++){
        for (int j=0; j< HISTORY_LEN ; j++){
            stacked.segment(idx, term_size[i]) = _history[j].segment(term_starts[i], term_size[i]);
            idx += term_size[i];
        }
    }
    return stacked;
}