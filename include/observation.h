#pragma once
#ifndef _OBSERVATION_H
#define _OBSERVATION_H

#include <eigen3/Eigen/Dense>
#include <deque>
#include <mujoco/mujoco.h>

class Observation
{
public:
    Observation();
    ~Observation();

    void setMujocoModel(const mjModel* m, mjData* d);
    void setVelocityCommand(double vx, double vy, double wz);
    // ? 함수 뒤에 붙는 const : 객체 안의 멤버변수들을 수정 못하게
    // ? 매개변수 앞에 붙는 const : 입력받은 인자를 수정하지 못하게 
    Eigen::VectorXd update(
        const Eigen::VectorXd& q,
        const Eigen::VectorXd& qdot,
        const Eigen::VectorXd& q_home,
        const Eigen::VectorXd& last_action
    );

    void reset();
private:
    const mjModel* _mj_model;
    mjData* _mj_data;
    int _pelvis_body_id;
    
    Eigen::VectorXd computeSingleObs(
        const Eigen::VectorXd& q,
        const Eigen::VectorXd& qdot,
        const Eigen::VectorXd& q_home,
        const Eigen::VectorXd& last_action
    );

    // ? 컴파일할때 값이 결정되어 종료될때까지 값을 유지 static: 모든 객체가 하나의 값을 공유, constexpr : 명시
    static constexpr int SINGLE_DIM = 164;
    static constexpr int HISTORY_LEN = 5;
    static constexpr int LEG_DIM = 14;
    static constexpr int NUM_BODIES = 32;
    static constexpr int STACKED_DIM = 820;

    std::deque<Eigen::VectorXd> _history;
    Eigen::Vector3d _vel_cmd;
    Eigen::Vector3d _default_com;
};

#endif