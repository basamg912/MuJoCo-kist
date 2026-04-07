#pragma once
#ifndef __MODEL_H
#define __MODEL_H

#include <iostream>
#include <eigen3/Eigen/Dense>
// ! 4/7 추가 RBDL -> Mujoco 
#include <mujoco/mujoco.h>
// #include <rbdl/rbdl.h>
// #include <rbdl/addons/urdfreader/urdfreader.h>
#include "custommath.h"\


using namespace std;
using namespace Eigen;

class CModel
{
public:
	CModel();
    // 가상 함수 테이블 확인 후 상속하는 클래스 소멸자도 모두 실행, (최하위 -> 최상위), 
    // 문제상황 CModel 타입 포인터로 CModel 을 상속하는 자식 클래스를 접근
    // CModel* model = new ChildModel(); -> delete model; -> CModel 소멸자 실행 -> ChildModel 소멸자 실행 X
    // -> 자식 클래스에서 할당한 메모리 해제 X -> 메모리 누수 발생 => 최상위 부모 클래스에서 소멸자 virtual 선언
	virtual ~CModel(); 

    // RigidBodyDynamics::Model _model;
    void set_mujoco_model(const mjModel* m, mjData* d);
    void update_kinematics(VectorXd & q, VectorXd & qdot); // update robot state
    void update_dynamics(); // calculate _A, _g, _b, _bg
    void calculate_EE_Jacobians(); // calcule jacobian
    void calculate_EE_positions_orientations(); // calculte End-effector postion, orientation
    void calculate_EE_velocity(); // calculate End-effector velocity

    MatrixXd _A; // inertia matrix
    VectorXd _g; // gravity force vector
	VectorXd _b; // Coriolis/centrifugal force vector
	VectorXd _bg; // Coriolis/centrifugal force vector + gravity force vector

    MatrixXd _J_hand; // jacobian Matrix 6x7
    MatrixXd _J_tmp; 

    Vector3d _position_local_task_hand; // End-effector coordinate
    Vector3d _tmp_position_local_task_hand;
    Vector3d _x_hand; // End-effector position
    Matrix3d _R_hand; // End-effector rotation matrix

    VectorXd _xdot_hand;

    VectorXd _max_joint_torque, _min_joint_torque, _max_joint_velocity, _min_joint_velocity, _max_joint_position, _min_joint_position;

private:
	void Initialize();
    // ! 4/7 RBDL -> Mujoco 
	// void load_model(); // read URDF model
	void set_robot_config();

    VectorXd _q, _qdot; // joint sensordata
    VectorXd _zero_vec_joint; // zero joint vector

    int _k; // joint number
    int _id_hand; // hand id

    // ! 4/7 RBDL -> Mujoco
    const mjModel* _mj_model;
    mjData* _mj_data;
    bool _bool_model_update, _bool_kinematics_update, _bool_dynamics_update, _bool_Jacobian_update; // update check

};

#endif