#include "controller.h"

CController::CController()
{
	_k = 31; // for kapex
	Initialize();
}

CController::~CController()
{
}

// ! free joint 가상관절. pelvis 가 시뮬레이션 상에서 좌표가 고정되어있지않으니 freejoint 로 설정
// ! 나머지 관절들은 좌표계가 부모 링크 기준이라서 고정
void CController::set_default_pose(mjData* d){

	int offset = Model.get_qpos_offset();

	if (offset == 7){
		d->qpos[0] = 0.0; d->qpos[1] = 0.0; d->qpos[2] = 1.30; d->qpos[3]=1.0; d->qpos[4]=0.0, d->qpos[5]=0.0; d->qpos[6]=0.0;
	}
	
	for (int i= offset; i< Model.getMjModel()->nq; i++){
		d->qpos[i] = 0.0;
	}

	// ! 하체
    d->qpos[offset + 2]  = -0.035;   // LLJ3
    d->qpos[offset + 3]  = -0.05;     // LLJ4 og : -0.38
    d->qpos[offset + 4]  = -0.33;    // LLJ5
    d->qpos[offset + 9]  = 0.035;    // RLJ3
    d->qpos[offset + 10] = 0.05;    // RLJ4 og : 0.38
    d->qpos[offset + 11] = 0.33;     // RLJ5

	// ! 팔
    d->qpos[offset + 17] = 0.2;      // LAJ1
    d->qpos[offset + 18] = 0.2;      // LAJ2
    d->qpos[offset + 19] = 0.18;     // LAJ3
    d->qpos[offset + 20] = -1.0;     // LAJ4
    d->qpos[offset + 24] = -0.2;     // RAJ1
    d->qpos[offset + 25] = -0.2;     // RAJ2
    d->qpos[offset + 26] = -0.18;    // RAJ3
    d->qpos[offset + 27] = 1.0;      // RAJ4
}

void CController::read(double t, double* q, double* qdot)
{	
	_t = t;
	if (_bool_init == true)
	{
		_init_t = _t;
		_bool_init = false;
	}

	_dt = t - _pre_t;
	// cout<<"_dt : "<<_dt<<endl;
	_pre_t = t;

	for (int i = 0; i < _k; i++)
	{
		_q(i) = q[i+ Model.get_qpos_offset()]; // ! free joint 7개 xyz 쿼터니언4개
		_qdot(i) = qdot[i+ Model.get_qvel_offset()]; // ! free joint 6개 xyz rpy
		// _qdot(i) = CustomMath::VelLowpassFilter(0.001, 2.0*PI* 10.0, _pre_q(i), _q(i), _pre_qdot(i)); //low-pass filter
		_pre_q(i) = _q(i);
		_pre_qdot(i) = _qdot(i);		
		if(_t < 2.0)///use filtered data after convergece
        {
			_qdot(i) = qdot[i+ Model.get_qvel_offset()];
		}
	}
}

void CController::write(double* ctrl)
{
	// position actuator: ctrl에 토크 대신 목표 위치를 전달
	for (int i = 0; i < _k; i++)
	{
		ctrl[i] = _q_des(i);
	}
}

void CController::control_mujoco()
{
    ModelUpdate(); // ! 동역학 계산

	Eigen::VectorXd stacked_obs = _obs.update(_q, _qdot, _q_home, _last_action);
	Eigen::VectorXd action = _policy->inference(stacked_obs);
	_last_action = action;

	_q_des = _q_home + 0.25 *action;
	_qdot_des.setZero();
	// ! position 제어 시 pd 제어는 무조코가 진행
	JointControl();

    // motionPlan(); // ! 목표 설정


	// if(_control_mode == 1) //joint space control
	// {
	// 	if (_t - _init_t < 0.1 && _bool_joint_motion == false)
	// 	{
	// 		_start_time = _init_t;
	// 		_end_time = _start_time + _motion_time;
	// 		JointTrajectory.reset_initial(_start_time, _q, _qdot);
	// 		JointTrajectory.update_goal(_q_goal, _qdot_goal, _end_time);
	// 		_bool_joint_motion = true;
	// 	}
		
	// 	JointTrajectory.update_time(_t);
	// 	_q_des = JointTrajectory.position_cubicSpline();
	// 	_qdot_des = JointTrajectory.velocity_cubicSpline();

	// 	JointControl();

	// 	if (JointTrajectory.check_trajectory_complete() == 1)
	// 	{
	// 		// _bool_plan(_cnt_plan) = 1;
	// 		_bool_init = true;
	// 	}
	// }
	// else if(_control_mode == 2)
	// {		
	// 	if (_t - _init_t < 0.1 && _bool_ee_motion == false)
	// 	{
	// 		_start_time = _init_t;
	// 		_end_time = _start_time + _motion_time;
	// 		HandTrajectory.reset_initial(_start_time, _x_hand, _xdot_hand);
	// 		HandTrajectory.update_goal(_x_goal_hand, _xdot_goal_hand, _end_time);
	// 		_bool_ee_motion = true;
	// 		// cout<<"_t : "<<_t<<endl;
	// 		// cout<<"_x_hand 	: "<<_x_hand.transpose()<<endl;
	// 	}

		
	// 	HandTrajectory.update_time(_t);
	// 	_x_des_hand.head(3) = HandTrajectory.position_cubicSpline();
	// 	_R_des_hand = HandTrajectory.rotationCubic();
	// 	_x_des_hand.segment<3>(3) = CustomMath::GetBodyRotationAngle(_R_des_hand);
	// 	_xdot_des_hand.head(3) = HandTrajectory.velocity_cubicSpline();
	// 	_xdot_des_hand.segment<3>(3) = HandTrajectory.rotationCubicDot();		

	// 	CLIK();

	// 	if (HandTrajectory.check_trajectory_complete() == 1)
	// 	{
	// 		_bool_plan(_cnt_plan) = 1;
	// 		_bool_init = true;
	// 	}
	// }
	
}

void CController::ModelUpdate()
{
    Model.update_kinematics(_q, _qdot);
	Model.update_dynamics();
    Model.calculate_EE_Jacobians();
	Model.calculate_EE_positions_orientations();
	Model.calculate_EE_velocity();

	_J_hands = Model._J_hand;

	_x_hand.head(3) = Model._x_hand;
	_x_hand.tail(3) = CustomMath::GetBodyRotationAngle(Model._R_hand);
	// cout << _x_hand.transpose() << endl;
	// cout << "R hand " << Model._R_hand.transpose() << endl;
	_xdot_hand = Model._xdot_hand;
}	

void CController::motionPlan()
{	
	if (_bool_plan(_cnt_plan) == 1)
	{
		if(_cnt_plan == 0)
		{	
			// cout << "plan: " << _cnt_plan << endl;
			// ! eigen 벡터 원소 접근 (일반 벡터로 치면 v[0])
			for (int i=0; i<_k; i++){
				_q_order(i) = _q_home(i);
			}
			// cout << "_q_home: "<< _q_home.transpose() << '\n';
			// cout << "_q_home: "<< _q.transpose() << '\n';  
			reset_target(10.0, _q_order, _qdot);
			_cnt_plan++;
		}
	}
}

void CController::reset_target(double motion_time, VectorXd target_joint_position)
{
	_control_mode = 3;
	_motion_time = motion_time;
	_bool_joint_motion = false;
	_bool_ee_motion = false;

	// _q_goal = target_joint_position.head(7);
	// _qdot_goal.setZero();
}

void CController::reset_target(double motion_time, VectorXd target_joint_position, VectorXd target_joint_velocity)
{
	_control_mode = 1;
	_motion_time = motion_time;
	_bool_joint_motion = false;
	_bool_ee_motion = false;

	_q_goal = target_joint_position.head(_k);
	// _qdot_goal = target_joint_velocity.head(7);
	_qdot_goal.setZero();
}

void CController::reset_target(double motion_time, Vector3d target_pos, Vector3d target_ori)
{
	_control_mode = 2;
	_motion_time = motion_time;
	_bool_joint_motion = false;
	_bool_ee_motion = false;

	_x_goal_hand.head(3) = target_pos;
	_x_goal_hand.tail(3) = target_ori;
	_xdot_goal_hand.setZero();
}

void CController::JointControl()
{
	// _torque.setZero();
	// _A_diagonal = Model._A;

	// // ! armature 질량행렬=관성
	// // ! 대각원소에 1을 더해서 정규화, 대각값이 너무 작은 관절이 있으면 토크 불안정하기 때문
	// for(int i = 0; i < _k; i++){
	// 	_A_diagonal(i,i) += 1.0;
	// }
	// // ! Model._bg : 중력보상


	// _torque = _A_diagonal*(_kpj*(_q_des - _q) + _kdj*(_qdot_des - _qdot)) + Model._bg;

	// _torque = _kp_diag.cwiseProduct(_q_des - _q) + _kd_diag.cwiseProduct(_qdot_des - _qdot) + Model._bg;

	// position actuator 사용: PD 계산은 MuJoCo가 처리
	// _q_des는 control_mujoco()에서 trajectory로 업데이트됨
	// write()에서 _q_des를 d->ctrl로 전달


}

// ! closed loop inverse kinematic
// ! xyz,orientation 을 관절각도로 변환
void CController::CLIK()
{
	_torque.setZero();	

	_x_err_hand.segment(0,3) = _x_des_hand.head(3) - _x_hand.head(3);
	_x_err_hand.segment(3,3) = -CustomMath::getPhi(Model._R_hand, _R_des_hand);

	_J_bar_hands = CustomMath::pseudoInverseQR(_J_hands);

	_dt = 0.003;
	_qdot_des = _J_bar_hands*(_xdot_des_hand + _x_kp*(_x_err_hand));
	_q_des = _q_des + _dt*_qdot_des;
	_A_diagonal = Model._A;
	for(int i = 0; i < 7; i++){
		_A_diagonal(i,i) += 1.0;
	}
	// _torque(0) = 400*(_q_des(0)-_q(0)) + 20*(_qdot_des(0)-_qdot(0));
	// _torque(1) = 2500*(_q_des(1)-_q(1)) + 250*(_qdot_des(1)-_qdot(1));
	// _torque(2) = 1500*(_q_des(2)-_q(2)) + 170*(_qdot_des(2)-_qdot(2));
	// _torque(3) = 1700*(_q_des(3)-_q(3)) + 320*(_qdot_des(3)-_qdot(3));
	// _torque(4) = 700*(_q_des(4)-_q(4)) + 70*(_qdot_des(4)-_qdot(4));
	// _torque(5) = 500*(_q_des(5)-_q(5)) + 50*(_qdot_des(5)-_qdot(5));
	// _torque(6) = 520*(_q_des(6)-_q(6)) + 15*(_qdot_des(6)-_qdot(6));

	_torque = _A_diagonal * (_kpj * (_q_des - _q) + _kdj * (_qdot_des - _qdot)) + Model._bg;

}

void CController::Initialize()
{
    _control_mode = 1; //1: joint space, 2: task space(CLIK)

	_bool_init = true;
	_t = 0.0;
	_init_t = 0.0;
	_pre_t = 0.0;
	_dt = 0.0;

	_kpj = 300.0;
	_kdj = 40.0;

	// _kpj_diagonal.setZero(_k, _k);
	// //							0 		1	2		3	   4	5 	6
	// _kpj_diagonal.diagonal() << 400., 2500., 1500., 1700., 700., 500., 520.;
	// _kdj_diagonal.setZero(_k, _k);
	// _kdj_diagonal.diagonal() << 20., 250., 170., 320., 70., 50., 15.;
	_x_kp = 1;//작게 0.1
	// _x_kp = 20.0;

    _q.setZero(_k);
	_qdot.setZero(_k);
	_torque.setZero(_k);

	_J_hands.setZero(6,_k);
	_J_bar_hands.setZero(_k,6);

	_x_hand.setZero(6);
	_xdot_hand.setZero(6);

	//////////////////원본///////////////////
	// _cnt_plan = 0;
	_bool_plan.setZero(30);
	// _time_plan.resize(30);
	// _time_plan.setConstant(5.0);
	//////////////////원본///////////////////

	_q_home.setZero(_k);
	// ! 
	// LL (인덱스 0~6)
	_q_home(2) = -0.035;   // LLJ3
	_q_home(3) = -0.38;     // LLJ4
	_q_home(4) = -0.33;    // LLJ5

	// RL (인덱스 7~13)
	_q_home(9)  = 0.035;   // RLJ3
	_q_home(10) = 0.38;   // RLJ4
	_q_home(11) = 0.33;    // RLJ5

	// WL (인덱스 14~16)
	// 전부 0

	// LA (인덱스 17~23)
	_q_home(17) = 0.2;     // LAJ1
	_q_home(18) = 0.2;     // LAJ2
	_q_home(19) = 0.18;    // LAJ3
	_q_home(20) = -1.0;    // LAJ4

	// RA (인덱스 24~30)
	_q_home(24) = -0.2;    // RAJ1
	_q_home(25) = -0.2;    // RAJ2
	_q_home(26) = -0.18;   // RAJ3
	_q_home(27) = 1.0;     // RAJ4

	_start_time = 0.0;
	_end_time = 0.0;
	_motion_time = 0.0;

	_bool_joint_motion = false;
	_bool_ee_motion = false;

	_q_des.setZero(_k);
	// ! 발산 방지
	_q_des = _q_home;

	_qdot_des.setZero(_k);
	_q_goal.setZero(_k);
	_qdot_goal.setZero(_k);

	_x_des_hand.setZero(6);
	_xdot_des_hand.setZero(6);
	_x_goal_hand.setZero(6);
	_xdot_goal_hand.setZero(6);

	_pos_goal_hand.setZero(); // 3x1 
	_rpy_goal_hand.setZero(); // 3x1
	JointTrajectory.set_size(_k);
	_A_diagonal.setZero(_k,_k);

	_x_err_hand.setZero(6);
	_R_des_hand.setZero();

	_I.setIdentity(_k,_k);

	_pre_q.setZero(_k);
	_pre_qdot.setZero(_k);

	///////////////////save_stack/////////////////////
	_q_order.setZero(_k);
	_qdot_order.setZero(_k);
	// _max_joint_position.setZero(7);
	// _min_joint_position.setZero(7);

	// _min_joint_position(0) = -2.9671;
	// _min_joint_position(1) = -1.8326;
	// _min_joint_position(2) = -2.9671;
	// _min_joint_position(3) = -3.1416;
	// _min_joint_position(4) = -2.9671;
	// _min_joint_position(5) = -0.0873;
	// _min_joint_position(6) = -2.9671;

	// _max_joint_position(0) = 2.9671;
	// _max_joint_position(1) = 1.8326;
	// _max_joint_position(2) = 2.9671;
	// _max_joint_position(3) = 0.0;
	// _max_joint_position(4) = 2.9671;
	// _max_joint_position(5) = 3.8223;
	// _max_joint_position(6) = 2.9671;

	///////////////////estimate_lr/////////////////////

	// cout << fixed;
	// cout.precision(3);
	_cnt_plan = 0;
	_bool_plan(_cnt_plan) = 1;

	_kp_diag.setZero(_k);
	_kd_diag.setZero(_k);
	
	// LL
	_kp_diag(0) = 83.83;   _kd_diag(0) = 10.67;   // LLJ1
	_kp_diag(1) = 83.83;   _kd_diag(1) = 10.67;   // LLJ2
	_kp_diag(2) = 277.15;  _kd_diag(2) = 20.0;    // LLJ3
	_kp_diag(3) = 277.15;  _kd_diag(3) = 20.0;    // LLJ4
	_kp_diag(4) = 40.0;    _kd_diag(4) = 2.0;     // LLJ5
	_kp_diag(5) = 40.0;    _kd_diag(5) = 2.0;     // LLJ6
	_kp_diag(6) = 5.0;     _kd_diag(6) = 0.05;    // LLJ7

	// RL (동일)
	_kp_diag(7) = 83.83;   _kd_diag(7) = 10.67;
	_kp_diag(8) = 83.83;   _kd_diag(8) = 10.67;
	_kp_diag(9) = 277.15;  _kd_diag(9) = 20.0;
	_kp_diag(10) = 277.15; _kd_diag(10) = 20.0;
	_kp_diag(11) = 40.0;   _kd_diag(11) = 2.0;
	_kp_diag(12) = 40.0;   _kd_diag(12) = 2.0;
	_kp_diag(13) = 5.0;    _kd_diag(13) = 0.05;

	// WL
	_kp_diag(14) = 200.0;  _kd_diag(14) = 20.0;
	_kp_diag(15) = 200.0;  _kd_diag(15) = 20.0;
	_kp_diag(16) = 200.0;  _kd_diag(16) = 20.0;

	// LA (왼팔) - 어깨 관절 더 강하게
	_kp_diag(17) = 200.0;  _kd_diag(17) = 15.0;   // LAJ1 (어깨 yaw)
	_kp_diag(18) = 200.0;  _kd_diag(18) = 15.0;   // LAJ2 (어깨 pitch)
	_kp_diag(19) = 100.0;  _kd_diag(19) = 8.0;    // LAJ3
	_kp_diag(20) = 100.0;  _kd_diag(20) = 8.0;    // LAJ4 (팔꿈치)
	_kp_diag(21) = 50.0;   _kd_diag(21) = 5.0;
	_kp_diag(22) = 50.0;   _kd_diag(22) = 5.0;
	_kp_diag(23) = 30.0;   _kd_diag(23) = 3.0;

	// RA - 동일
	_kp_diag(24) = 200.0;  _kd_diag(24) = 15.0;
	_kp_diag(25) = 200.0;  _kd_diag(25) = 15.0;
	_kp_diag(26) = 100.0;  _kd_diag(26) = 8.0;
	_kp_diag(27) = 100.0;  _kd_diag(27) = 8.0;
	_kp_diag(28) = 50.0;   _kd_diag(28) = 5.0;
	_kp_diag(29) = 50.0;   _kd_diag(29) = 5.0;
	_kp_diag(30) = 30.0;   _kd_diag(30) = 3.0;
}