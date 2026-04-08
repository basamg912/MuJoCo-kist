#include "JointControl.h"
#include "timer.h"
#include "StateDef.h"

#include <chrono>
#include <cmath>
#include <vector>
#include <stdio.h>


#define toe_l1 0.0842882
#define toe_l2 0.045
#define toe_l3 0.0850000
#define toe_l4 0.0450000
#define toe_l1_ 0.1208344
#define toe_l2_ 0.018
#define toe_l3_ 0.120
#define toe_l4_ 0.012
#define knee_l1 0.2050299
#define knee_l2 0.0675 
#define knee_l3 0.188
#define knee_l4 0.0685

#define toe_offset1 1.72538013864403
#define toe_offset41 1.70939117736651
#define toe_offset_del 0.979304243294018
#define toe_offset_link  2.3619331327729
#define toe_offset_input2 1.50974645488939
#define toe_offset42 1.56118479860416
#define knee_offset 0.843074313858853

// #define elbow_l1 74.0/1000.0
// #define elbow_l2 21.0/1000.0
// #define elbow_l3 74.0/1000.0
// #define elbow_l4 21.0/1000.0
// #define elbow_offset 130.0/180.0*3.14159265358979

#define wrist_l1 12.0/1000.0
#define wrist_l2 10.5/1000.0
#define wrist_l3 29.69/1000.0
#define wrist_l4 51.0/1000.0
#define wrist_l5 7.3/1000.0
#define wrist_l6 15.0/1000.0
#define wrist_dx 9.5/1000.0
#define wrist_dy 35.45/1000.0
#define wrist_dz 31.5/1000.0

using namespace std;
using namespace Eigen;

CJointControl::CJointControl()
{
	Initialize();
	m_mlp = MLPs(2, 2);
	m_mlp.load_weights("/home/kist/Bak/weight_bias_25Aug2025_2/");
}

CJointControl::~CJointControl()
{

}

void CJointControl::ConvertStateMotor2Joint(int nState, double dTime, double *dTheta, double *dThetaDot)
{
	// HOMING : 제어시작
	// CONTROL : 동작시작
	// STANDSTILL : 댐핑
	// cout<<"nState : "<<nState<<endl;
	m_iControlMode = nState;
	if(m_iControlMode != m_iPreviousControlMode){
		if (m_iControlMode == INIT) {
			m_dInitStartTime = dTime;
			m_bFirstLoop = true;

			m_bINIT = true;
			m_bHOMING = false;
			m_bCONTROL = false;
			m_bRL = false;
		}
		else if (m_iControlMode == HOMING) {
			m_dHomingStartTime = dTime;
			m_bFirstLoop = true;

			m_bINIT = false;
			m_bHOMING = true;
			m_bCONTROL = false;
			m_bRL = false;
		}
		else if (m_iControlMode == CONTROL) {
			m_dControlStartTime = dTime;
			m_bFirstLoop = true;

			m_bINIT = false;
			m_bHOMING = false;
			m_bCONTROL = true;
			m_bRL = false;
		}
		else if (m_iControlMode == RL){
			m_dRLStartTime = dTime;
			m_bFirstLoop = true;

			m_bINIT = false;
			m_bHOMING = false;
			m_bCONTROL = false;
			m_bRL = true;
		}
		else if (m_iControlMode == STANDSTILL) {
			m_bTorqueOff = true;
		}
		m_iPreviousControlMode = m_iControlMode;
	}
	if(m_bTorqueOff){
			m_bINIT = false;
			m_bHOMING = false;
			m_bCONTROL = false;
			m_bRL = false;
	}

	// save motor data
    for(int i=0; i<m_nJDoF; i++)
    {
        m_vMotorThetarad(i) = dTheta[i];
        m_vMotorThetadotrad(i) = dThetaDot[i];
    }

	// save real time
	m_dTime = dTime;
	m_dInitTime = m_dTime - m_dInitStartTime;
	m_dHomingTime = m_dTime - m_dHomingStartTime;
	m_dControlTime = m_dTime - m_dControlStartTime;
	m_dRLTime = m_dTime - m_dRLStartTime;

	// m_rtTimeStart = read_timer();
	// m_mlp.forward(m_vMotorThetarad.segment(4,2));
	// m_rtTimeEnd = read_timer();
	// m_vComputationTime(1) = (m_rtTimeEnd - m_rtTimeStart)/ 1000000.0;
	// cout<<"time: "<<m_vComputationTime(1)<<endl;

	// convert motor state to joint state
	m_vJointQrad = psi2q(m_vMotorThetarad);

	// save first joint position for initial position
	if(m_bFirstLoop){
		m_bFirstLoop = false;
		m_bjointmotion = true;
		m_bmotormotion = true;
		m_vPreJointQradDes = m_vJointQradDes;
		m_vPreJointQdotradDes = m_vJointQdotradDes;
	}

	// calculate Motor2Joint Jacobian
	m_mActuation_tau_q2psi = compute_tau_q2psi(m_vMotorThetarad);

	m_vJpsidot = m_mActuation_tau_q2psi * m_vMotorThetadotrad;

	// LPF qdot (Not used now) -> Please set _dt correctly in the low pass filter
	m_vJpsidot_filtered(0) = m_vJpsidot(0);
	m_vJpsidot_filtered(1) = m_vJpsidot(1);
	m_vJpsidot_filtered(2) = m_vJpsidot(2);
	m_vJpsidot_filtered(3) = m_vJpsidot(3);
	m_vJpsidot_filtered(4) = CustomMath::LowPassFilter(0.001, 100.0, m_vJpsidot(4), m_vJpsidot_filtered(4));
	m_vJpsidot_filtered(5) = CustomMath::LowPassFilter(0.001, 100.0, m_vJpsidot(5), m_vJpsidot_filtered(5));
	m_vJpsidot_filtered(6) = CustomMath::LowPassFilter(0.001, 100.0, m_vJpsidot(6), m_vJpsidot_filtered(6));

	m_vJpsidot_filtered(7) = m_vJpsidot(7);
	m_vJpsidot_filtered(8) = m_vJpsidot(8);
	m_vJpsidot_filtered(9) = m_vJpsidot(9);
	m_vJpsidot_filtered(10) = m_vJpsidot(10);
	m_vJpsidot_filtered(11) = CustomMath::LowPassFilter(0.001, 100.0, m_vJpsidot(11), m_vJpsidot_filtered(11));
	m_vJpsidot_filtered(12) = CustomMath::LowPassFilter(0.001, 100.0, m_vJpsidot(12), m_vJpsidot_filtered(12));
	m_vJpsidot_filtered(13) = CustomMath::LowPassFilter(0.001, 100.0, m_vJpsidot(13), m_vJpsidot_filtered(13));

	m_vJpsidot_filtered(14) = m_vJpsidot(14);
	m_vJpsidot_filtered(15) = m_vJpsidot(15);
	m_vJpsidot_filtered(16) = m_vJpsidot(16);

	m_vJpsidot_filtered.segment(17,33) = m_vJpsidot.segment(17,33);
	m_vJointQdotrad = m_vJpsidot_filtered;

	// Raw qdot version -> either use this, or comment this line and uncomment LPF qdot version
	// m_vJointQdotrad = m_vJpsidot;

	// debug
    // if(m_nCoutNum == 100){
    //     cout<<"JOINT : "<<m_vJointQrad.transpose()<<endl;
    //     cout<<"MOTOR : "<<m_vMotorThetarad.transpose()<<endl;
    //     cout<<"JOINT DOT : "<<m_vJointQdotrad.transpose()<<endl;
    //     cout<<"MOTOR DOT : "<<m_vMotorThetadotrad.transpose()<<endl;
    //     cout<<"___________________________________________"<<endl;
    //     m_nCoutNum = 0;
    // }
    // m_nCoutNum ++;
	// cout<<"start ModelUpdate"<<endl;
	ModelUpdate();
	// cout<<"check ModelUpdate"<<endl;
	m_rtTimeStart = read_timer();
	if(m_bMPPItimeout){
		m_dMPPItimeoutstep = 0;
	}
	m_bMPPItimeout = true;
	// m_dNu = pow(m_mKpj(0,0), 2);
	// m_dLambda = m_mKdj(0,0);
	// m_mR(0,0) = m_mKpj(1,1);
	// m_mR(1,1) = m_mKpj(1,1);
	// m_dGamma = m_mKdj(1,1);
	// m_dist = std::normal_distribution<double>(0.0, sqrt(m_dNu));
	Compute();
	// cout<<"check Compute"<<endl;
	m_rtTimeEnd = read_timer();
	m_vComputationTime(0) = (m_rtTimeEnd - m_rtTimeStart)/ 1000000000.0;
	// cout<<m_vComputationTime(0)<<endl;
	// cout<<"check Compute"<<endl;
}

void CJointControl::ConvertCmdJoint2Motor(double *dTarget)
{
	//for debug -> you should off this for tau on
	// m_vTorque.setZero();
	// m_vTorque(6) = 0.0;

	// safe mode
	if(m_bINIT || m_bHOMING || m_bCONTROL || m_bTorqueOff || m_bRL){
		for(int i=0;i<m_nJDoF;i++){
			if(abs(m_vTorque(i)) > m_vTorqueLimitNm(i) && !m_bRL){
				m_bTorqueOff = true;
				std::cout << "\033[1;34m"; 
				std::cout << "=======================\n";
				std::cout << "   " << m_dTime << "\n";
				std::cout << "i: " << i << "\n";
				std::cout << "MOTOR TORQUE LIMIT :" << ", "<< m_vTorque(i) << ", " <<m_vTorqueLimitNm(i) << "\n";
				std::cout << "\033[0m"; 
				std::cout<<endl;
			}
			// if(abs(m_vJointQdotrad(i)) > m_vJointQdotradLimit(i)){
			// 	m_nCoutVelLimitNum ++;
			// 	if(m_nCoutVelLimitNum > 10){
			// 		m_bTorqueOff = true;
			// 	}
			// 	std::cout << "\033[1;34m"; 
			// 	std::cout << "=======================\n";
			// 	std::cout << "m_dTime   " << m_dTime << "\n";
			// 	std::cout << "i: " << i << "\n";
			// 	std::cout << "JOINT VEL LIMIT:" << ", "<< m_vJointQdotrad(i)  << ", " <<m_vJointQdotradLimit(i) << "\n";
			// 	std::cout << "\033[0m"; 
			// 	std::cout<<endl;
			// }
			if(abs(m_vMotorThetadotrad(i)) > m_vMotorThetadotradLimit(i)){
				m_nCoutVelLimitNum ++;
				if(m_nCoutVelLimitNum > 10){
					m_bTorqueOff = true;
				}
				std::cout << "\033[1;34m"; 
				std::cout << "=======================\n";
				std::cout << "m_dTime   " << m_dTime << "\n";
				std::cout << "i: " << i << "\n";
				std::cout << "MOTOR VEL LIMIT:" << ", "<< m_vMotorThetadotrad(i) << ", " <<m_vMotorThetadotradLimit(i) << "\n";
				std::cout << "\033[0m"; 
				std::cout<<endl;
			}
			// if(m_vJointQrad(i) > m_vJointQradUpLimit(i)){
			// 	m_bTorqueOff = true;
			// 	std::cout << "\033[1;34m"; 
			// 	std::cout << "=======================\n";
			// 	std::cout << "m_dTime   " << m_dTime << "\n";
			// 	std::cout << "i: " << i << "\n";
			// 	std::cout << "JOINT UPPER POSITION LIMIT:" << ", "<< m_vJointQrad(i)  << ", " <<m_vJointQradUpLimit(i) << "\n";
			// 	std::cout << "\033[0m"; 
			// 	std::cout<<endl;
			// }
			// if(m_vJointQrad(i) < m_vJointQradLowLimit(i)){
			// 	m_bTorqueOff = true;
			// 	std::cout << "\033[1;34m"; 
			// 	std::cout << "=======================\n";
			// 	std::cout << "m_dTime   " << m_dTime << "\n";
			// 	std::cout << "i: " << i << "\n";
			// 	std::cout << "JOINT LOWER POSITION LIMIT:" << ", "<< m_vJointQrad(i)  << ", " <<m_vJointQradLowLimit(i) << "\n";
			// 	std::cout << "\033[0m"; 
			// 	std::cout<<endl;
			// }
			// if(m_vMotorThetarad(i) > m_vMotorThetaradUpLimit(i)){
			// 	m_bTorqueOff = true;
			// 	std::cout << "\033[1;34m"; 
			// 	std::cout << "=======================\n";
			// 	std::cout << "m_dTime   " << m_dTime << "\n";
			// 	std::cout << "i: " << i << "\n";
			// 	std::cout << "MOTOR UPPER POSITION LIMIT:" << ", "<< m_vMotorThetarad(i)  << ", " <<m_vMotorThetaradUpLimit(i) << "\n";
			// 	std::cout << "\033[0m"; 
			// 	std::cout<<endl;
			// }
			// if(m_vMotorThetarad(i) < m_vMotorThetaradLowLimit(i)){
			// 	m_bTorqueOff = true;
			// 	std::cout << "\033[1;34m"; 
			// 	std::cout << "=======================\n";
			// 	std::cout << "m_dTime   " << m_dTime << "\n";
			// 	std::cout << "i: " << i << "\n";
			// 	std::cout << "MOTOR LOWER POSITION LIMIT:" << ", "<< m_vMotorThetarad(i)  << ", " <<m_vMotorThetaradLowLimit(i) << "\n";
			// 	std::cout << "\033[0m"; 
			// 	std::cout<<endl;
			// }
		}
	}

	if(m_dMPPItimeoutstep > 3){
		// std::cout << "\033[1;34m"; 
		// cout << "=======================\n";
		// cout <<"TIME LIMIT"<<endl;
		// cout << "   " << m_dTime << "\n\n\n";
		// std::cout << "\033[0m"; 
		m_bTorqueOff = true;
	}
	

	if(m_bTorqueOff){
		// m_vTorque = (m_mKdj*(- m_vMotorThetadotrad));

		// m_vTorque.segment(0,17) = (1.0*m_mKdj*(- m_vMotorThetadotrad)).segment(0,17);
		// m_vRLTorque.segment(0,17) = (1.0*m_mKdj*(- m_vMotorThetadotrad)).segment(0,17);

		m_vTorque.setZero();
		m_vRLTorque.setZero();
	}

	// m_vTorque.setZero();
	
	if(m_bINIT || m_bHOMING || m_bCONTROL || m_bTorqueOff || m_bRL){
		for(int i=0;i<m_nJDoF;i++){
			m_vTorqueLPF(i) = m_vTorque(i);
			if(m_bRL){
				m_vTorqueLPF(i) = m_vRLTorque(i);
			}
			// RL LPF
			m_vTorqueLPF(i) = CustomMath::LowPassFilter(0.001, 500.0*2*3.14159265358979, m_vTorqueLPF(i), m_vTorqueLPF_Pre(i));
			m_vTorqueLPF_Pre(i) = m_vTorqueLPF(i);


			dTarget[i] = m_vTorqueLPF(i);

			if(m_dTime <= 1.0){
				dTarget[i] = 0.0;
			}

			// safe mode
			// dTarget[i] = 0.0;
		}
	}
	// if(m_bTorqueOff){
	// 	for(int i=0;i<m_nJDoF;i++){
	// 		dTarget[i] = 0.0;
	// 	}
	// }
}

void CJointControl::ModelUpdate(){
	// cout<<"check update_modelID"<<endl;
	m_modelRobot.update_kinematics(m_vJointQrad, m_vJointQdotrad);
	// cout<<"check update_kinematics"<<endl;
	m_modelRobot.update_dynamics();
	// cout<<"check update_dynamics"<<endl;
	m_modelRobot.calculate_EE_Jacobians();
	// cout<<"check calculate_EE_Jacobians"<<endl;
	m_modelRobot.calculate_EE_positions_orientations();
	// cout<<"check calculate_EE_positions_orientations"<<endl;
	m_modelRobot.calculate_EE_velocity();
	// cout<<"check calculate_EE_velocity"<<endl;
	// cout<<"check RBDL"<<endl;

	m_vX_left_heel.head(3) = m_modelRobot.m_x_left_heel;
	m_vX_left_heel.tail(3) = CustomMath::GetBodyRotationAngle(m_modelRobot.m_R_left_heel);
	m_vX_left_toe.head(3) = m_modelRobot.m_x_left_toe;
	m_vX_left_toe.tail(3) = CustomMath::GetBodyRotationAngle(m_modelRobot.m_R_left_toe);
	// m_vXdot_left_heel = m_modelRobot.m_xdot_left_heel;
	// m_vXdot_left_toe = m_modelRobot.m_xdot_left_toe;
	// m_mJ_left_heel = m_modelRobot.m_J_left_heel;
	// m_mJ_left_toe = m_modelRobot.m_J_left_toe;
}

void CJointControl::Compute(){
	if(m_bINIT){
		// Set cubic desired q, qdot target for Init
		if (m_bmotormotion == true)
		{
			// cout<<"start homing! : "<<m_dHomingTime<<endl;
			m_dmotiontime = 2.0;

			m_vmotorgoal.setZero();
			m_vmotordotgoal.setZero();

			m_dmotionendtime = m_dInitTime + m_dmotiontime-0.001;
			MotorTrajectory.reset_initial(m_dInitTime, m_vMotorThetarad, m_vMotorThetadotrad);
			MotorTrajectory.update_goal(m_vmotorgoal, m_vmotordotgoal, m_dmotionendtime);

			m_bmotormotion = false;
			m_bINITCOMPLETE = false;			
		}
		
		MotorTrajectory.update_time(m_dInitTime);
		m_vMotorThetaradDes = MotorTrajectory.position_cubicSpline();
		m_vMotorThetadotradDes = MotorTrajectory.velocity_cubicSpline();

		for(int i = 0; i<m_nJDoF; i++){
			if(abs(m_vMotorThetarad(i) - m_vmotorgoal(i))<0.001){
				m_vMotorThetaradDes(i) = 0.0;
				m_vMotorThetadotradDes(i) = 0.0;
			}
		}
		if(MotorTrajectory.check_trajectory_complete() == 1){
			m_bINITCOMPLETE = true;
		}
	}
	else if(m_bHOMING){
		// Set cubic desired q, qdot target for homing
		if (m_bjointmotion == true)
		{
			// cout<<"start homing! : "<<m_dHomingTime<<endl;
			m_dmotiontime = 2.0;

			m_vjointgoal.setZero();
			// 자세 종류 테스트 시 사용
			// m_vjointgoal.segment(0,m_nJDoF) << -0.0, 0.2, -0.0, +0.0, -0.0, 0.0, 0.0,
			// 									0.0, -0.00, +.0, -0.0, +0.0, 0.0, 0.0,
			// 									-0.0, 0.0, 0.0; 	
			// RL offseet 변경 (여기서 수정하면 됨 @@@@@)
			m_vjointgoal.segment(0,m_nJDoF) << -0.05, 0.05, -0.07, +0.38, -0.33, 0.0, 0.101,
												0.05, -0.05, +0.07, -0.38, +0.33, 0.0, -0.101,
												0.0, 0.0, -0.05;
			// 0점 자세
			// m_vjointgoal.segment(0,m_nJDoF) << 0.0, 0.0, -0.0, 0.0, -0.0, 0.0, 0.0,
			// 								0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0,
			// 								0.0, 0.0, 0.0;
			m_vjointdotgoal.setZero();

			m_dmotionendtime = m_dHomingTime + m_dmotiontime-0.0005;
			JointTrajectory.reset_initial(m_dHomingTime, m_vJointQrad, m_vJointQdotrad);
			JointTrajectory.update_goal(m_vjointgoal, m_vjointdotgoal, m_dmotionendtime);

			m_bjointmotion = false;
			
		}
		
		JointTrajectory.update_time(m_dHomingTime);
		m_vJointQradDes = JointTrajectory.position_cubicSpline();
		m_vJointQdotradDes = JointTrajectory.velocity_cubicSpline();

		for(int i = 0; i<m_nJDoF; i++){
			if(abs(m_vJointQrad(i) - m_vjointgoal(i))<0.001){
				m_vJointQradDes(i) = m_vjointgoal(i);
				m_vJointQdotradDes(i) = 0.0;
			}
		}

		m_vRLJointQradDes = m_vJointQradDes;
		m_vRLJointQradDesLPF_Pre = m_vJointQradDes;
	}

	else if(m_bCONTROL){
		// Set sin desired q, qdot targetv
		m_vJointQradDes.segment(0,m_nJDoF) <<
			-0.0*sin(2*(m_dControlTime))*sin(2*(m_dControlTime)),
			0.0*sin(1.5*(m_dControlTime)), 
			-0.0*sin(1.0*(m_dControlTime))*sin(1.0*(m_dControlTime)), 
			0.0*sin(1.0*(m_dControlTime))*sin(1.0*(m_dControlTime)), 
			0.0*sin(2.5*(m_dControlTime)), 
			0.0*sin(2.0*(m_dControlTime)),
			0.0*sin(2*(m_dControlTime)),

			0.0*sin(2*(m_dControlTime))*sin(2*(m_dControlTime)),
			-0.0*sin(1.5*(m_dControlTime)), 
			0.0*sin(1.0*(m_dControlTime))*sin(1.0*(m_dControlTime)), 
			-0.0*sin(3*(m_dControlTime))*sin(3*(m_dControlTime)), 
			-0.0*sin(2.5*(m_dControlTime)), 
			0.0*sin(2.0*(m_dControlTime)), 
			-0.0*sin(2*(m_dControlTime)),

			0.0*sin(1.0*(m_dControlTime)),
			0.0*sin(1.0*(m_dControlTime)),
			0.2*sin(1.0*(m_dControlTime))*sin(1.0*(m_dControlTime));

			//사실 오른팔임
			// 0.2*sin(2.5*(m_dControlTime)),
			// -0.3*sin(1.5*(m_dControlTime))*sin(1.5*(m_dControlTime)),
			// 0.25*sin(2.0*(m_dControlTime))*sin(2.0*(m_dControlTime)),
			// 0.3*sin(2.5*(m_dControlTime))*sin(2.5*(m_dControlTime)),
			// 0.25*sin(3*(m_dControlTime))*sin(3*(m_dControlTime)),
			// 5.0*sin(1.0*(m_dControlTime)),
			// -5.0*sin(1.0*(m_dControlTime)),

			// 0.15*sin(2.5*(m_dControlTime)),
			// -0.15*sin(1.5*(m_dControlTime))*sin(1.5*(m_dControlTime)),
			// 0.15*sin(2*(m_dControlTime))*sin(2*(m_dControlTime)),
			// 0.15*sin(2.5*(m_dControlTime))*sin(2.5*(m_dControlTime)),
			// 0.15*sin(3*(m_dControlTime))*sin(3*(m_dControlTime)),
			// -0.3*sin(4.0*(m_dControlTime)),
			// -0.3*sin(4.0*(m_dControlTime))

		m_vJointQradDes += m_vPreJointQradDes;
		m_vJointQdotradDes.setZero();

		// Set cubic desired q, qdot target
		// if (m_bjointmotion == false)
		// {
		// 	cout<<"cubic motion! : "<<m_dHomingTime<<endl;
		// 	m_dmotiontime = 10.0;

		// 	m_vjointgoal.setZero();
		// 	m_vjointgoal.segment(0,m_nJDoF) << 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0,
		// 									0.0, 0.0, 0.0, -0.1, 0.0, 0.0, 0.0,
		// 									0.0, 0.0, 1.0,
		// 									0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		// 									0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
		// 	m_vjointdotgoal.setZero();

		// 	m_dmotionendtime = m_dHomingTime + m_dmotiontime-0.001;
		// 	JointTrajectory.reset_initial(m_dHomingTime, m_vPreJointQradDes, m_vPreJointQdotradDes);
		// 	JointTrajectory.update_goal(m_vjointgoal, m_vjointdotgoal, m_dmotionendtime);

		// 	m_bjointmotion = true;
		// }
		
		// JointTrajectory.update_time(m_dHomingTime);
		// m_vJointQradDes = JointTrajectory.position_cubicSpline();
		// m_vJointQdotradDes = JointTrajectory.velocity_cubicSpline();

		// for(int i = 0; i<m_nJDoF; i++){
		// 	if(abs(m_vInitJointQrad(i) - m_vjointgoal(i))<0.0174533){
		// 		m_vJointQradDes(i) = m_vjointgoal(i);
		// 		m_vJointQdotradDes(i) = 0.0;
		// 	}
		// }	

	}

	// m_vTorque.setZero();
	// RL des 반영 하는 곳(@@@@@)

	// waist home자세와 통일
	m_vRLJointQradDes.segment(14,3) = m_vJointQradDes.segment(14,3);

	// hip offset
	// m_vJointQradDes(7) += 0.0175*0.5;
	// m_vJointQradDes(8) -= 0.0175*0.5;

	// m_vRLJointQradDes(0) -= 0.0175*0.5;
	// m_vRLJointQradDes(1) += 0.0175*0.5;
	// m_vRLJointQradDes(7) += 0.0175*0.5;
	// m_vRLJointQradDes(8) -= 0.0175*0.5;

	// toe offset
	// m_vRLJointQradDes(6) = 0.0;
	// m_vRLJointQradDes(13) = -0.0;
	m_vRLJointQdotradDes.setZero();

	// RL qdes LPF
	// for(int i = 0; i < m_nJDoF; i++){
	// 	m_vRLJointQradDes(i) = CustomMath::LowPassFilter(0.001, 10.0*2*3.14159265358979, m_vRLJointQradDes(i), m_vRLJointQradDesLPF_Pre(i));
	// 	m_vRLJointQradDesLPF_Pre(i) = m_vRLJointQradDes(i);
	// }

	m_mKpRLj.block<3,3>(14,14) = m_mKpj.block<3,3>(14,14);
	m_mKdRLj.block<3,3>(14,14) = m_mKdj.block<3,3>(14,14);

	// Joint Task
	if(m_bINIT){
		m_vTorque = (m_mKpj*(m_vMotorThetaradDes - m_vMotorThetarad) + m_mKdj*(m_vMotorThetadotradDes - m_vMotorThetadotrad));
		if(m_bINITCOMPLETE){
			m_vTorque.setZero();
		}
	}
	else if(m_bHOMING || m_bCONTROL || m_bRL){
		m_vTorque = m_mActuation_tau_q2psi.transpose()*(m_mKpj*(m_vJointQradDes - m_vJointQrad) + m_mKdj*(m_vJointQdotradDes - m_vJointQdotrad));
		// m_vTorque = m_mActuation_tau_q2psi.transpose()*(m_mKpRLj*(m_vJointQradDes - m_vJointQrad) + m_mKdRLj*(m_vJointQdotradDes - m_vJointQdotrad));

		m_vRLTorque = m_mKpRLj*(m_vRLJointQradDes - m_vJointQrad) + m_mKdRLj*(m_vRLJointQdotradDes - m_vJointQdotrad);
		// cout<<"m_vRLTorque "<<m_vRLTorque<<endl;
		for(int i = 0; i < m_nJDoF; i++){
			if (abs(m_vRLTorque(i)) > m_vJointTorqueLimitNm(i)){
				m_vRLTorque(i) = std::clamp(m_vRLTorque(i), -m_vJointTorqueLimitNm(i), m_vJointTorqueLimitNm(i));
			}
		}
		m_vRLTorque = m_mActuation_tau_q2psi.transpose()*m_vRLTorque;
			
		// toe tau offset (@@@@@)
		// m_vTorque(6) = 3;
		// m_vTorque(13) = -3;
	}

	// debug
	// m_nCoutNum=0;
	if(m_nCoutNum == 100){
		if(!m_bTorqueOff){
			cout<<"control mode : "<<m_iControlMode<<endl;
			cout<<"joint"<<endl;
			cout<<m_vJointQrad.segment(0,7).transpose()<<endl;
			cout<<m_vJointQrad.segment(7,7).transpose()<<endl;
			cout<<m_vJointQrad.segment(14,3).transpose()<<endl;
			// cout<<m_vJointQrad.segment(17,7).transpose()<<endl;
			// cout<<m_vJointQrad.segment(24,7).transpose()<<endl;
			cout<<"jointdot"<<endl;
			cout<<m_vJointQdotrad.segment(0,7).transpose()<<endl;
			cout<<m_vJointQdotrad.segment(7,7).transpose()<<endl;
			cout<<m_vJointQdotrad.segment(14,3).transpose()<<endl;
			// cout<<m_vJointQdotrad.segment(17,7).transpose()<<endl;
			// cout<<m_vJointQdotrad.segment(24,7).transpose()<<endl<<endl;

			cout<<"motor"<<endl;
			cout<<m_vMotorThetarad.segment(0,7).transpose()<<endl;
			cout<<m_vMotorThetarad.segment(7,7).transpose()<<endl;
			cout<<m_vMotorThetarad.segment(14,3).transpose()<<endl;
			// cout<<m_vMotorThetarad.segment(17,7).transpose()<<endl;
			// cout<<m_vMotorThetarad.segment(24,7).transpose()<<endl;
			cout<<"motordot"<<endl;
			cout<<m_vMotorThetadotrad.segment(0,7).transpose()<<endl;
			cout<<m_vMotorThetadotrad.segment(7,7).transpose()<<endl;
			cout<<m_vMotorThetadotrad.segment(14,3).transpose()<<endl;
			// cout<<m_vMotorThetadotrad.segment(17,7).transpose()<<endl;
			// cout<<m_vMotorThetadotrad.segment(24,7).transpose()<<endl<<endl;

			cout<<"m_vTorque"<<endl;
			cout<<m_vTorque.segment(0,7).transpose()<<endl;
			cout<<m_vTorque.segment(7,7).transpose()<<endl;
			cout<<m_vTorque.segment(14,3).transpose()<<endl;
			// cout<<m_vTorque.segment(17,7).transpose()<<endl;
			// cout<<m_vTorque.segment(24,7).transpose()<<endl<<endl;
			
			cout<<"RL joint des"<<endl;
			cout<<m_vRLJointQradDes.segment(0,7).transpose()<<endl;
			cout<<m_vRLJointQradDes.segment(7,7).transpose()<<endl;
			cout<<m_vRLJointQradDes.segment(14,3).transpose()<<endl;

			cout<<"m_vRLTorque"<<endl;
			cout<<m_vRLTorque.segment(0,7).transpose()<<endl;
			cout<<m_vRLTorque.segment(7,7).transpose()<<endl;
			cout<<m_vRLTorque.segment(14,3).transpose()<<endl;


			// cout<<"RL action"<<endl;
			// for(int i = 0; i < 12; i++){
			// 	m_vRLJactions(i) = m_pcRLControl-> m_vRLactions(i);
			// }
			// cout<<m_vRLJactions.segment(0,6).transpose()<<endl;
			// cout<<m_vRLJactions.segment(6,6).transpose()<<endl;

			// cout<<"RL observation"<<endl;
			// for(int i = 0; i < 47; i++){
			// 	m_vRLJobservations(i) = m_pcRLControl-> m_vRLobservations(i);
			// }
			// cout<<m_vRLJobservations.transpose()<<endl;

			cout<<"___________________________________________"<<endl;
		}
		if(m_bTorqueOff){
			cout << "\033[1;34m"; 
			cout<<"Damping mode!"<<endl;
			cout << "\033[0m"; 
		}

		m_nCoutNum = 0;
	}
	m_nCoutNum ++;

	// m_vTorque(0) = 0.0;	//hip yaw
	// m_vTorque(1) = 0.0;	//hip roll
	// m_vTorque(2) = 0.0;	//hip pitch
	// m_vTorque(3) = 0.0;	//knee
	// m_vTorque(4) = 0.0;	//ankle1
	// m_vTorque(5) = 0.0;	//ankle2
	// m_vTorque(6) = 0.0;	//toe

	// m_vTorque(7) = 0.0;	//hip yaw
	// m_vTorque(8) = 0.0;	//hip roll
	// m_vTorque(9) = 0.0;	//hip pitch
	// m_vTorque(10) = 0.0;	//knee
	// m_vTorque(11) = 0.0;	//ankle1
	// m_vTorque(12) = 0.0;	//ankle2
	// m_vTorque(13) = 0.0;	//toe

	// m_vTorque(14) = 0.0;	//waist roll
	// m_vTorque(15) = 0.0;	//waist 1
	// m_vTorque(16) = 0.0;	//waist 2

	// 사실 right arm임
	m_vTorque(17) = 0.0; //left shoulder pitch
	m_vTorque(18) = 0.0; //left shoulder roll
	m_vTorque(19) = 0.0; //left shoulder yaw
	m_vTorque(20) = 0.0; //left elbow
	m_vTorque(21) = 0.0; //left wrist 1
	m_vTorque(22) = 0.0; //left wrist 2
	m_vTorque(23) = 0.0; //left wrist yaw

	//여기아래로 절대 터치금지
	m_vTorque(24) = 0.0; //right shoulder pitch
	m_vTorque(25) = 0.0; //right shoulder roll
	m_vTorque(26) = 0.0; //right shoulder yaw
	m_vTorque(27) = 0.0; //right elbow
	m_vTorque(28) = 0.0; //right wrist 1
	m_vTorque(29) = 0.0; //right wrist 2
	m_vTorque(30) = 0.0; //right wrist yaw
}

Vector<double,50> CJointControl::psi2q(Vector<double, 50> vPsi)
{
	m_vQ_psi2q.setZero();
    // left leg
    m_vQ_psi2q(0) = vPsi(0);
    m_vQ_psi2q(1) = vPsi(1);
    m_vQ_psi2q(2) = vPsi(2);

	m_dTheta2_psi2q = knee_offset + vPsi(3);
	m_dL_squared_psi2q = knee_l1*knee_l1 + knee_l2*knee_l2 -2*knee_l1*knee_l2*cos(M_PI - m_dTheta2_psi2q); 
	m_dBeta_psi2q = acos((knee_l1*knee_l1 + m_dL_squared_psi2q - knee_l2*knee_l2)/(2*knee_l1*sqrt(m_dL_squared_psi2q)));
	m_dLambda_psi2q = acos((m_dL_squared_psi2q + knee_l4*knee_l4 - knee_l3*knee_l3)/(2*knee_l4*sqrt(m_dL_squared_psi2q)));

	m_vQ_psi2q(3) = m_dLambda_psi2q - m_dBeta_psi2q;

	m_vQ_psi2q.segment(4,2) = m_mlp.forward(vPsi.segment(4,2));

	m_dTheta1_psi2q = toe_offset1 + vPsi(6);
	m_dL_squared_psi2q = toe_l1*toe_l1 + toe_l2*toe_l2 - 2*toe_l1*toe_l2*cos(m_dTheta1_psi2q);
	m_dLambda1_psi2q = acos((m_dL_squared_psi2q + toe_l4*toe_l4 - toe_l3*toe_l3)/(2*sqrt(m_dL_squared_psi2q)*toe_l4));
	m_dBeta1_psi2q = acos((toe_l1*toe_l1 + m_dL_squared_psi2q - toe_l2*toe_l2)/(2*toe_l1*sqrt(m_dL_squared_psi2q)));
	m_dTheta41_psi2q = M_PI - m_dBeta1_psi2q - m_dLambda1_psi2q;

	m_dTheta2_psi2q = 2*M_PI - (toe_offset_link + m_vQ_psi2q(4)) - m_dBeta1_psi2q - m_dLambda1_psi2q - toe_offset_del;
	m_dL__squared_psi2q = toe_l1_*toe_l1_ + toe_l2_*toe_l2_ - 2*toe_l1_*toe_l2_*cos(m_dTheta2_psi2q);
	m_dLambda2_psi2q = acos((m_dL__squared_psi2q + toe_l4_*toe_l4_ - toe_l3_*toe_l3_)/(2*sqrt(m_dL__squared_psi2q)*toe_l4_));
	m_dBeta2_psi2q = acos((toe_l1_*toe_l1_ + m_dL__squared_psi2q - toe_l2_*toe_l2_)/(2*toe_l1_*sqrt(m_dL__squared_psi2q)));
	m_dTheta42_psi2q = M_PI - m_dBeta2_psi2q - m_dLambda2_psi2q;

	if(!std::isnan(m_dTheta42_psi2q)){
		m_vQ_psi2q(6) = m_dTheta42_psi2q - toe_offset42;
	}
	else{
		m_vQ_psi2q(6) = 0.0;
	}

	// right leg
    m_vQ_psi2q(0+7) = vPsi(0+7);
    m_vQ_psi2q(1+7) = vPsi(1+7);
    m_vQ_psi2q(2+7) = vPsi(2+7);

	vPsi.segment(7,7) = -vPsi.segment(7,7);

	m_dTheta2_psi2q = knee_offset + vPsi(3+7);
	m_dL_squared_psi2q = knee_l1*knee_l1 + knee_l2*knee_l2 -2*knee_l1*knee_l2*cos(M_PI - m_dTheta2_psi2q); 
	m_dBeta_psi2q = acos((knee_l1*knee_l1 + m_dL_squared_psi2q - knee_l2*knee_l2)/(2*knee_l1*sqrt(m_dL_squared_psi2q)));
	m_dLambda_psi2q = acos((m_dL_squared_psi2q + knee_l4*knee_l4 - knee_l3*knee_l3)/(2*knee_l4*sqrt(m_dL_squared_psi2q)));

	m_vQ_psi2q(3+7) = m_dLambda_psi2q - m_dBeta_psi2q;

	//mlp
	m_vQ_psi2q.segment(4+7,2) = m_mlp.forward(vPsi.segment(4+7,2));

	m_dTheta1_psi2q = toe_offset1 + vPsi(6+7);
	m_dL_squared_psi2q = toe_l1*toe_l1 + toe_l2*toe_l2 - 2*toe_l1*toe_l2*cos(m_dTheta1_psi2q);
	m_dLambda1_psi2q = acos((m_dL_squared_psi2q + toe_l4*toe_l4 - toe_l3*toe_l3)/(2*sqrt(m_dL_squared_psi2q)*toe_l4));
	m_dBeta1_psi2q = acos((toe_l1*toe_l1 + m_dL_squared_psi2q - toe_l2*toe_l2)/(2*toe_l1*sqrt(m_dL_squared_psi2q)));
	m_dTheta41_psi2q = M_PI - m_dBeta1_psi2q - m_dLambda1_psi2q;


	m_dTheta2_psi2q = 2*M_PI - (toe_offset_link + m_vQ_psi2q(4+7)) - m_dBeta1_psi2q - m_dLambda1_psi2q - toe_offset_del;
	m_dL__squared_psi2q = toe_l1_*toe_l1_ + toe_l2_*toe_l2_ - 2*toe_l1_*toe_l2_*cos(m_dTheta2_psi2q);
	m_dLambda2_psi2q = acos((m_dL__squared_psi2q + toe_l4_*toe_l4_ - toe_l3_*toe_l3_)/(2*sqrt(m_dL__squared_psi2q)*toe_l4_));
	m_dBeta2_psi2q = acos((toe_l1_*toe_l1_ + m_dL__squared_psi2q - toe_l2_*toe_l2_)/(2*toe_l1_*sqrt(m_dL__squared_psi2q)));
	m_dTheta42_psi2q = M_PI - m_dBeta2_psi2q - m_dLambda2_psi2q;

	if(!std::isnan(m_dTheta42_psi2q)){
		m_vQ_psi2q(6+7) = m_dTheta42_psi2q - toe_offset42;
	}
	else{
		m_vQ_psi2q(6+7) = 0.0;
	}

	m_vQ_psi2q(3+7) = -m_vQ_psi2q(3+7);
	m_vQ_psi2q(4+7) = -m_vQ_psi2q(4+7);
	m_vQ_psi2q(5+7) = -m_vQ_psi2q(5+7);
	m_vQ_psi2q(6+7) = -m_vQ_psi2q(6+7);

	// waist
	//roll
	m_vQ_psi2q(14) = vPsi(14);
	//yaw
	m_vQ_psi2q(15) = (vPsi(15) + vPsi(16));
	//ptich
	m_vQ_psi2q(16) = (vPsi(15) - vPsi(16))*0.5;

	//head가 중간에 낌~ urdf도 이게 맞음 뒤에 +2될 예정!!

	// 일단 모터 기준임 포워드 풀어서 반영해야함!!!
	//left arm
	m_vQ_psi2q(17) = vPsi(17);	//shoulder pitch
	m_vQ_psi2q(18) = vPsi(18);	//shoulder roll
	m_vQ_psi2q(19) = vPsi(19);	//shoulder yaw

	m_vQ_psi2q(20) = vPsi(20);	//elbow

	//vPsi : wrist 1, 2, yaw -> m_vQ_psi2q : wrist yaw roll pitch
	m_vQ_psi2q(21) = vPsi(23);	//wrist yaw
	m_vQ_psi2q(22) = vPsi(21);	//wrist 1
	m_vQ_psi2q(23) = vPsi(22);
	
	//right arm
	m_vQ_psi2q(24) = vPsi(24);	//shoulder pitch
	m_vQ_psi2q(25) = vPsi(25);	//shoulder roll
	m_vQ_psi2q(26) = vPsi(26);	//shoulder yaw

	m_vQ_psi2q(27) = vPsi(27);	//elbow

	//vPsi : wrist 1, 2, yaw -> m_vQ_psi2q : wrist yaw roll pitch
	m_vQ_psi2q(28) = vPsi(30);
	m_vQ_psi2q(29) = vPsi(28);
	m_vQ_psi2q(30) = vPsi(29);

	return m_vQ_psi2q; 
}

Vector<double, 4> CJointControl::compute_wrist_q2psi(Vector<double,2> vLeftwristq, Vector<double,2> vRightwristq){
    // --- 왼쪽 손목 계산 ---
    s1 = sin(vLeftwristq(0));
    c1 = cos(vLeftwristq(0));
    s2 = sin(vLeftwristq(1));
    c2 = cos(vLeftwristq(1));

    DeltaX1 = -wrist_dx + (wrist_l2 - wrist_l5) * c1 * s2 + wrist_l6 * c2;
    K_1 = wrist_dy + wrist_l1 * c1 + ((wrist_l2 - wrist_l5) * c1 + wrist_l3 * s1) * c2 - wrist_l6 * s2;
    DeltaZ1 = wrist_dz + wrist_l1 * s1 + (wrist_l2 - wrist_l5) * s1 * c2 - wrist_l3 * c1 - wrist_l6 * s1 * s2;

    DeltaX2 = wrist_dx + (wrist_l2 - wrist_l5) * c1 * s2 - wrist_l6 * c2;
    K_2 = wrist_dy + wrist_l1 * c1 + ((wrist_l2 - wrist_l5) * c1 + wrist_l3 * s1) * c2 + wrist_l6 * s2;
    DeltaZ2 = wrist_dz + wrist_l1 * s1 + (wrist_l2 - wrist_l5) * s1 * c2 - wrist_l3 * c1 + wrist_l6 * s1 * s2;

    m_vPsi_q2psi(0) = K_1 - sqrt(wrist_l4 * wrist_l4 - DeltaX1 * DeltaX1 - DeltaZ1 * DeltaZ1);
    m_vPsi_q2psi(1) = K_2 - sqrt(wrist_l4 * wrist_l4 - DeltaX2 * DeltaX2 - DeltaZ2 * DeltaZ2);

    // --- 오른쪽 손목 계산 ---
    s1 = sin(vRightwristq(0));
    c1 = cos(vRightwristq(0));
    s2 = sin(vRightwristq(1));
    c2 = cos(vRightwristq(1));

    DeltaX1 = -wrist_dx + (wrist_l2 - wrist_l5) * c1 * s2 + wrist_l6 * c2;
    K_1 = wrist_dy + wrist_l1 * c1 + ((wrist_l2 - wrist_l5) * c1 + wrist_l3 * s1) * c2 - wrist_l6 * s2;
    DeltaZ1 = wrist_dz + wrist_l1 * s1 + (wrist_l2 - wrist_l5) * s1 * c2 - wrist_l3 * c1 - wrist_l6 * s1 * s2;

    DeltaX2 = wrist_dx + (wrist_l2 - wrist_l5) * c1 * s2 - wrist_l6 * c2;
    K_2 = wrist_dy + wrist_l1 * c1 + ((wrist_l2 - wrist_l5) * c1 + wrist_l3 * s1) * c2 + wrist_l6 * s2;
    DeltaZ2 = wrist_dz + wrist_l1 * s1 + (wrist_l2 - wrist_l5) * s1 * c2 - wrist_l3 * c1 + wrist_l6 * s1 * s2;

    m_vPsi_q2psi(2) = K_1 - sqrt(wrist_l4 * wrist_l4 - DeltaX1 * DeltaX1 - DeltaZ1 * DeltaZ1);
    m_vPsi_q2psi(3) = K_2 - sqrt(wrist_l4 * wrist_l4 - DeltaX2 * DeltaX2 - DeltaZ2 * DeltaZ2);

    return m_vPsi_q2psi;

	// 손목 지그로 0점 잡고, 손목 abs 받아오기, 리스트 조인트 값에 abs 넣어주고, 모터 값에 ik푼거 던저주기.
}


Vector2d CJointControl::state_transition(Vector2d vState, Vector2d vControl, double dDelta_t){
	return (m_vF + m_mG*vControl)*dDelta_t;
}

// long double CJointControl::running_cost(Vector2d vState, Vector2d vControl, Vector2d vControlVar){
// 	m_vPsi << 0.0, 0.0, 0.0, 0.0, vState.segment(0,2), 0.0;
// 	m_vQ = psi2qleft(m_vPsi).segment(4,2);

// 	if ((m_vQ - m_vQlb).minCoeff() < 0.0 || (m_vQ - m_vQub).maxCoeff() > 0.0) {
// 		m_dIndicatorFunc = 1.0;
// 	}
// 	else{
// 		m_dIndicatorFunc = 0.0;
// 	}

// 	m_ldQx = (m_vJointQradDes.segment(4,2) - m_vQ).transpose()*m_mWq*(m_vJointQradDes.segment(4,2) - m_vQ);// + 1.0e+6*m_dIndicatorFunc;

// 	return m_ldQx + 0.5*(1.0 - 1.0/m_dNu)*vControlVar.transpose()*m_mR*vControlVar + vControl.transpose()*m_mR*vControlVar + 0.5*vControl.transpose()*m_mR*vControl;
// }

// long double CJointControl::terminal_cost(Vector2d vState){
// 	m_vPsi << 0.0, 0.0, 0.0, 0.0, vState.segment(0,2), 0.0;
// 	m_vQ = psi2qleft(m_vPsi).segment(4,2);

// 	if ((m_vQ - m_vQlb).minCoeff() < 0.0 || (m_vQ - m_vQub).maxCoeff() > 0.0) {
// 		m_dIndicatorFunc = 1.0;
// 	}
// 	else{
// 		m_dIndicatorFunc = 0.0;
// 	}

// 	m_ldQx = (m_vJointQradDes.segment(4,2) - m_vQ).transpose()*m_mWq*(m_vJointQradDes.segment(4,2) - m_vQ);// + 1.0e+6*m_dIndicatorFunc;

// 	return m_ldQx;
// }


Eigen::Matrix<double, 50, 50> CJointControl::compute_tau_q2psi(Vector<double,50> vX){
	// left leg
    // Partial derivative w.r.t. psi(4)
	m_vEpsilon_jacobian.setZero();
    m_vEpsilon_jacobian(4) = 1e-5;
	m_vNum_jacobian = psi2q(vX + m_vEpsilon_jacobian) - psi2q(vX - m_vEpsilon_jacobian);
	m_vDenom_jacobian = 2*m_vEpsilon_jacobian;
	m_mJ_jacobian(4,4) = m_vNum_jacobian(4)/m_vDenom_jacobian(4);
	m_mJ_jacobian(5,4) = m_vNum_jacobian(5)/m_vDenom_jacobian(4);
	m_mJ_jacobian(6,4) = m_vNum_jacobian(6)/m_vDenom_jacobian(4);

    // Partial derivative w.r.t. psi(5)
	m_vEpsilon_jacobian.setZero();
    m_vEpsilon_jacobian(5) = 1e-5;
	m_vNum_jacobian = psi2q(vX + m_vEpsilon_jacobian) - psi2q(vX - m_vEpsilon_jacobian);
	m_vDenom_jacobian = 2*m_vEpsilon_jacobian;
	m_mJ_jacobian(4,5) = m_vNum_jacobian(4)/m_vDenom_jacobian(5);
	m_mJ_jacobian(5,5) = m_vNum_jacobian(5)/m_vDenom_jacobian(5);
	m_mJ_jacobian(6,5) = m_vNum_jacobian(6)/m_vDenom_jacobian(5);

    // Partial derivative w.r.t. psi(6)
	m_vEpsilon_jacobian.setZero();
    m_vEpsilon_jacobian(6) = 1e-5;
	m_vNum_jacobian = psi2q(vX + m_vEpsilon_jacobian) - psi2q(vX - m_vEpsilon_jacobian);
	m_vDenom_jacobian = 2*m_vEpsilon_jacobian;
	m_mJ_jacobian(6,6) = m_vNum_jacobian(6)/m_vDenom_jacobian(6);

    // Partial derivative w.r.t. psi(3)
	m_vEpsilon_jacobian.setZero();
    m_vEpsilon_jacobian(3) = 1e-5;
	m_vNum_jacobian = psi2q(vX + m_vEpsilon_jacobian) - psi2q(vX - m_vEpsilon_jacobian);
	m_vDenom_jacobian = 2*m_vEpsilon_jacobian;
	m_mJ_jacobian(3,3) = m_vNum_jacobian(3)/m_vDenom_jacobian(3);
	
	// right leg
    // Partial derivative w.r.t. psi(11)
	m_vEpsilon_jacobian.setZero();
    m_vEpsilon_jacobian(11) = 1e-5;
	m_vNum_jacobian = psi2q(vX + m_vEpsilon_jacobian) - psi2q(vX - m_vEpsilon_jacobian);
	m_vDenom_jacobian = 2*m_vEpsilon_jacobian;
	m_mJ_jacobian(11,11) = m_vNum_jacobian(11)/m_vDenom_jacobian(11);
	m_mJ_jacobian(12,11) = m_vNum_jacobian(12)/m_vDenom_jacobian(11);
	m_mJ_jacobian(13,11) = m_vNum_jacobian(13)/m_vDenom_jacobian(11);

    // Partial derivative w.r.t. psi(12)
	m_vEpsilon_jacobian.setZero();
    m_vEpsilon_jacobian(12) = 1e-5;
	m_vNum_jacobian = psi2q(vX + m_vEpsilon_jacobian) - psi2q(vX - m_vEpsilon_jacobian);
	m_vDenom_jacobian = 2*m_vEpsilon_jacobian;
	m_mJ_jacobian(11,12) = m_vNum_jacobian(11)/m_vDenom_jacobian(12);
	m_mJ_jacobian(12,12) = m_vNum_jacobian(12)/m_vDenom_jacobian(12);
	m_mJ_jacobian(13,12) = m_vNum_jacobian(13)/m_vDenom_jacobian(12);

    // Partial derivative w.r.t. psi(13)
	m_vEpsilon_jacobian.setZero();
    m_vEpsilon_jacobian(13) = 1e-5;
	m_vNum_jacobian = psi2q(vX + m_vEpsilon_jacobian) - psi2q(vX - m_vEpsilon_jacobian);
	m_vDenom_jacobian = 2*m_vEpsilon_jacobian;
	m_mJ_jacobian(13,13) = m_vNum_jacobian(13)/m_vDenom_jacobian(13);

    // Partial derivative w.r.t. psi(10)
	m_vEpsilon_jacobian.setZero();
    m_vEpsilon_jacobian(10) = 1e-5;
	m_vNum_jacobian = psi2q(vX + m_vEpsilon_jacobian) - psi2q(vX - m_vEpsilon_jacobian);
	m_vDenom_jacobian = 2*m_vEpsilon_jacobian;
	m_mJ_jacobian(10,10) = m_vNum_jacobian(10)/m_vDenom_jacobian(10);

	// waist roll yaw pitch
	m_mJ_jacobian(14, 14) = 1.0;
	m_mJ_jacobian(15, 15) = 1.0;
	m_mJ_jacobian(15, 16) = 1.0;
	m_mJ_jacobian(16, 15) = 0.5;
	m_mJ_jacobian(16, 16) = -0.5;

	// head

	// 일단 모터 기준임 포워드 풀어서 반영해야함!!!
	// left arm
	m_mJ_jacobian(21, 21) = 0.0;
	m_mJ_jacobian(22, 22) = 0.0;
	m_mJ_jacobian(23, 23) = 0.0;

	m_mJ_jacobian(21, 23) = 1.0;
	m_mJ_jacobian(22, 21) = 1.0;
	m_mJ_jacobian(23, 22) = 1.0;
	// right arm
	m_mJ_jacobian(28, 28) = 0.0;
	m_mJ_jacobian(29, 29) = 0.0;
	m_mJ_jacobian(30, 30) = 0.0;

	m_mJ_jacobian(28, 30) = 1.0;
	m_mJ_jacobian(29, 28) = 1.0;
	m_mJ_jacobian(30, 29) = 1.0;

	return m_mJ_jacobian;
}

void CJointControl::Initialize()
{	
	cout<<"Joint Control Initialize!"<<endl;
    /* public Robot State */
	m_vTorque.setZero();
	m_vMotorThetarad.setZero();
	m_vMotorThetaradDes.setZero();
	m_vMotorThetadotrad.setZero();
	m_vMotorThetadotradDes.setZero();
	m_vJointQrad.setZero();
	m_vJointQradDes.setZero();
	m_vPreJointQradDes.setZero();
	m_vPreJointQdotradDes.setZero();
	m_vJointQdotrad.setZero();
	m_vJointQdotradDes.setZero();

	m_vTorqueLPF.setZero();
	m_vTorqueLPF_Pre.setZero();
	m_vRLTorque.setZero();
	m_vRLJointQradDes.setZero();
	m_vRLJointQdotradDes.setZero();
	m_vRLJointQradDesLPF_Pre.setZero();
	m_vRLJactions.setZero();
	m_vRLJobservations.setZero();

	/* Robot State */
	m_dTime = 0.0;
	m_iControlMode = 0;
	m_iPreviousControlMode = 0;
	m_dInitStartTime = 0.0;
	m_dInitTime = 0.0;
	m_dHomingStartTime = 0.0;
	m_dHomingTime = 0.0;
	m_dControlStartTime = 0.0;
	m_dControlTime = 0.0;
	m_dRLStartTime = 0.0;
	m_dRLTime = 0.0;
	m_bINIT = false;
	m_bINITCOMPLETE = false;
	m_bHOMING = false;
	m_bCONTROL = false;
	m_bRL = false;
	m_bFirstLoop = true;

	/* Motor2Joint */
	m_mActuation_tau_q2psi.setZero();
	m_vJpsidot.setZero();
	m_vJpsidot_filtered.setZero();

	/* model update */
	m_bFisrtModelUpdate = true;
    m_mJ_left_heel.setZero();
	m_mJ_left_toe.setZero();
    m_vX_left_heel.setZero();
	m_vX_left_toe.setZero();
    m_vXdot_left_heel.setZero();
	m_vXdot_left_toe.setZero();
    m_vXdes_left_heel.setZero();
	m_vXdes_left_toe.setZero();
    m_vXdotdes_left_heel.setZero();
	m_vXdotdes_left_toe.setZero();

	/* PD gain */
	m_mKpj.setZero();
	m_mKdj.setZero();
	m_mKpRLj.setZero();
	m_mKdRLj.setZero();

	/* Joint Control */
	m_bjointmotion = true;
	m_bmotormotion = true;
	JointTrajectory.set_size(50);
	MotorTrajectory.set_size(50);
	m_dmotiontime, m_dmotionendtime = 0.0;
	m_vjointgoal.setZero();
	m_vjointdotgoal.setZero();
	m_vmotorgoal.setZero();
	m_vmotordotgoal.setZero();

	/* CLIK Control */
	m_btaskmotion = true;
	PositionTrajectory.set_size(3);
	RotationTrajectory.set_size(3);
	m_nCLIKerrGain = 0;
	m_vCLIKerr.setZero();
	m_vX_left_toe_init.setZero();
	m_mR_left_toe.setZero();
	m_mRdes_left_toe.setZero();
	m_vX_left_heel_init.setZero();
	m_mR_left_heel.setZero();
	m_mRdes_left_heel.setZero();
	m_vtaskgoal.setZero();
	m_vtaskdotgoal.setZero();

	/* public debug state */
	m_vComputationTime.setZero();

	/* Debug */
	m_nCoutNum = 0;
	m_nCoutVelLimitNum = 0;
	m_vTorqueLimitNm.setZero();
	m_vTorqueLimitNm.segment(0, m_nJDoF) << 70.0, 70.0, 150.0, 150.0, 70.0, 70.0, 20.0,
											70.0, 70.0, 150.0, 150.0, 70.0, 70.0, 20.0,
											120.0, 120.0, 120.0,
											70.0, 70.0, 30.0, 30.0, 0.5, 0.5, 20.0,
											70.0, 70.0, 30.0, 30.0, 0.5, 0.5, 20.0;

	m_vJointTorqueLimitNm.segment(0, m_nJDoF) << 70, 70, 180, 180, 60, 60, 20,
												 70, 70, 180, 180, 60, 60, 20,
												 120.0, 120.0, 120.0,
												 70.0, 70.0, 30.0, 30.0, 0.5, 0.5, 20.0,
												 70.0, 70.0, 30.0, 30.0, 0.5, 0.5, 20.0;
											
	m_vJointQradUpLimit.setZero();
	m_vJointQradUpLimit.segment(0, m_nJDoF) << 0.3491, 0.3491, 0.5236, 1.91986, 1.0123, 1.0123, 2.3491,
												0.7854, 0.3491, 1.7453, 0.175, 1.0123, 1.0123, 2.0472,
												0.523599, 0.523599, 1.5708,
												3.141, 0.261, 1.658, 1.658, 1.745, 28, 28,
												1.5, 1.5, 1.5, 2.0, 1.5, 1.5, 1.5;
	m_vJointQradLowLimit.setZero();
	m_vJointQradLowLimit.segment(0, m_nJDoF) << -0.7854, -0.3491, -1.7453, -0.175, -1.0123, -1.0123, -2.0472,
												-0.3491, -0.3491, -0.5236, -1.91986, -1.0123, -1.0123, -2.3491,
												-0.523599, -0.523599, -0.174533,
												-0.872, -3.054, -1.658, -0.349, -1.745, -13, -13,
												-1.5, -1.5, -1.5, -2.0, -1.5, -1.5, -1.5;

	m_vMotorThetaradUpLimit.setZero();
	m_vMotorThetaradUpLimit.segment(0, m_nJDoF) << 0.3491, 0.3491, 0.5236, 1.9111, 1.2741, 0.5, 0.7854,
													0.7854, 0.3491, 1.7453, 0.0175, 0.6109, 1.513, 1.1868,
													0.3491, 2.0944, 0.6981,
													3.141, 0.261, 1.658, 1.658, 28, 28, 1.745,
													1.5, 1.5, 1.5, 2.0, 1.5, 1.5, 1.5;

	m_vMotorThetaradLowLimit.setZero();
	m_vMotorThetaradLowLimit.segment(0, m_nJDoF) << -0.7854, -0.3491, -1.7453, -0.0175, -0.6109, -1.513,-1.1868,
													-0.3491, -0.3491, -0.5236, -1.9111, -1.2741, -0.5, -0.7854,
													-0.3491, -0.6981, -2.0944,
													-0.872, -3.054, -1.658, -0.349, -13, -13, -1.745,
													-1.5, -1.5, -1.5, -2.0, -1.5, -1.5, -1.5;

	// cout<<"m_vMotorThetaradLowLimit"<<m_vMotorThetaradLowLimit.transpose()<<endl;

	m_vJointQdotradLimit.setZero();
	m_vJointQdotradLimit.segment(0, m_nJDoF) << 10, 10, 10, 10, 30, 30, 50.0,
											 	  10, 10, 10, 10, 30, 30, 50.0,
											 	  5.0, 5.0, 5.0,
											 	  0.8, 0.8, 8.99, 5.0, 5.0, 30, 30,
											 	  0.8, 0.8, 8.99, 5.0, 5.0, 30, 30;
	m_vMotorThetadotradLimit.setZero();
	m_vMotorThetadotradLimit.segment(0, m_nJDoF) << 10.99, 10.99, 10.99, 10.99, 10.99, 10.99, 50.0,
											 	  	  10.99, 10.99, 10.99, 10.99, 10.99, 10.99, 50.0,
											 	  	  5.0, 5.0, 5.0,
											 	  	  0.8, 0.8, 1.5, 5.0, 30, 30, 5.0,
											 	  	  0.8, 0.8, 1.5, 5.0, 30, 30, 5.0;
													  
	m_bTorqueOff = false;
	m_dPreviousComputationTime = 0.0;
	m_bMPPItimeout = true;
	m_dMPPItimeoutstep = 0;
	m_ldMPPItimes = 0.0;

    /* Pre-allocated Variables for Functions */
    m_vQ_psi2q.setZero();
    m_dTheta2_psi2q = 0.0; m_dBeta_psi2q = 0.0; m_dLambda_psi2q = 0.0; m_dL_squared_psi2q = 0.0;
    m_dTheta1_psi2q = 0.0; m_dTheta41_psi2q = 0.0; m_dBeta1_psi2q = 0.0; m_dLambda1_psi2q = 0.0;
    m_dTheta42_psi2q = 0.0; m_dBeta2_psi2q = 0.0; m_dLambda2_psi2q = 0.0; m_dL__squared_psi2q = 0.0;

	m_vPsi_q2psi.setZero();
    
    m_mJ_jacobian.setZero();
	m_mJ_jacobian.setIdentity();
    m_vEpsilon_jacobian.setZero();
    m_vNum_jacobian.setZero();
    m_vDenom_jacobian.setZero();

    m_vPsi.setZero();
    m_vQ.setZero();
    m_dIndicatorFunc = 0.0;
    m_ldQx = 0.0L;

    m_ldDenominator_compute = 0.0L;

	/* MPPI state */
	m_nState = 2;
	m_nCtrl = 2;
	m_nK = 175;
	m_nN = 3;
	m_mU.setZero();
	m_mDelta_u.setZero();
	m_vF.setZero();
	m_mG.setZero();
	m_mG.setIdentity();
	m_dDel_t = 0.001;
	m_mX.setZero();
	m_vS.setZero();
	m_dNu = pow(3.0, 2); // step: 2.0, trajectory: 3.0
	m_mWq.setZero();
	m_mWq.diagonal() << 1800.0, 1000.0;
	m_mR.setZero();
	m_mR << 1.0e-3, 0.0,
		  0.0, 1.0e-3;
	m_dLambda = 0.2;	// step: 4.0, trajectory: 0.2
	m_dGamma = 0.99;	
	m_gen.seed(m_rd());
	m_dist = std::normal_distribution<double>(0.0, sqrt(m_dNu));
	m_vQlb.setZero();
	m_vQlb << -0.15, -0.15; 
	m_vQub.setZero();
	m_vQub << 0.15, 0.15;

	cout<<"Joint Control Initialized"<<endl;
}

void CJointControl::SetDesiredPosition(const std::array<float, 17>& qDes)
{
	for (size_t i = 0; i < qDes.size(); ++i) {
		m_vRLJointQradDes(i) = static_cast<double>(qDes[i]);
	}
}