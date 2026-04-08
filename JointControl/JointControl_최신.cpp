#include "JointControl.h"
#include "timer.h"


#include <chrono>
#include <cmath>
#include <vector>
#include <stdio.h>


#define toe_l1 0.13301456367328   
#define toe_l2 0.04499567507054    
#define toe_l3 0.13159532612761   
#define toe_l4 0.04499615661815   
#define toe_l1_ 0.1208354562901 
#define toe_l2_ 0.018  
#define toe_l3_ 0.11999116264078  
#define toe_l4_ 0.0119998955265

#define knee_l1 0.21500001081872  
#define knee_l2 0.0675   
#define knee_l3 0.199   
#define knee_l4 0.06850000830099  

#define toe_offset1 1.570414168037626   
#define toe_offset_del 0.979303534947913 
#define toe_offset_link 2.255082121300977 
#define toe_offset42 1.561131109001393 
#define knee_offset 0.802453535284488   

// #define elbow_l1 74.0/1000.0
// #define elbow_l2 21.0/1000.0
// #define elbow_l3 74.0/1000.0
// #define elbow_l4 21.0/1000.0
// #define elbow_offset 130.0/180.0*3.14159265358979

#define wrist_l1 12.0/1000.0
#define wrist_l2 10.5/1000.0
#define wrist_l3 30/1000.0
#define wrist_l4 48.5/1000.0
#define wrist_l5 11.88709298/1000.0
#define wrist_l6 12.0/1000.0
#define wrist_dx 9.5/1000.0
#define wrist_dy 36.42326083/1000.0
#define wrist_dz 31.5/1000.0

using namespace std;
using namespace Eigen;

CJointControl::CJointControl()
{
	cout<<"start CJointControl"<<endl;
	Initialize();
	m_mlp = MLPs(2, 2);
	m_mlp.load_weights("/home/kist/Bak/Bak/weight_bias_260119/");
	cout<<"loaded MLP weights"<<endl;
}

CJointControl::~CJointControl()
{

}

void CJointControl::precompute_trajectory() {
    if (q_trajectory_.empty() || target_indices_.empty()) {
        std::cerr << "\033[31m" << "[Warning] No trajectory data loaded." << "\033[0m" << std::endl;
        return;
    }

    q_interp_buffer_.clear();
    q_dot_interp_buffer_.clear();

    double control_dt = 0.001;

    int num_segments = q_trajectory_.size() - 1;
    int steps_per_segment = static_cast<int>(std::round(csv_dt_ / control_dt));
    int total_steps = num_segments * steps_per_segment + 1;

    q_interp_buffer_.reserve(total_steps);
    q_dot_interp_buffer_.reserve(total_steps);

    std::cout << "[Trajectory] Re-calculating smooth velocity & processing..." << std::endl;

    for (size_t i = 0; i < num_segments; ++i) {
        // 위치 csv 파일 데이터 그대로 사용
        Eigen::VectorXd q_start = q_trajectory_[i];
        Eigen::VectorXd q_end   = q_trajectory_[i+1];

        // 속도 csv 파일 데이터 사용 X
        Eigen::VectorXd v_start(q_start.size());
        Eigen::VectorXd v_end(q_end.size());

        if (i == 0) {
            v_start.setZero();
        } else {
            v_start = (q_trajectory_[i+1] - q_trajectory_[i-1]) / (2.0 * csv_dt_);
        }

        if (i == num_segments - 1) {
            v_end.setZero();
        } else {
            v_end = (q_trajectory_[i+2] - q_trajectory_[i]) / (2.0 * csv_dt_);
        }

        joint_traj_.reset_initial(0.0, q_start, v_start);
        joint_traj_.update_goal(q_end, v_end, csv_dt_);

        for (int k = 0; k < steps_per_segment; ++k) {
            double t_local = k * control_dt;
            joint_traj_.update_time(t_local);
            q_interp_buffer_.push_back(joint_traj_.position_cubicSpline());
            q_dot_interp_buffer_.push_back(joint_traj_.velocity_cubicSpline());
        }
    }

    q_interp_buffer_.push_back(q_trajectory_.back());
    // 마지막 속도는 0으로 설정
    q_dot_interp_buffer_.push_back(Eigen::VectorXd::Zero(target_indices_.size())); 

    std::cout << "[Trajectory] Done. Smooth Interpolation applied." << std::endl;
}

void CJointControl::load_segmented_data(const std::string& name_path, 
                                        const std::string& theta_path, 
                                        const std::string& vel_path,
                                        const std::vector<std::string>& target_joints) {
    std::vector<std::string> all_joint_names = read_csv_names(name_path);
    
    target_indices_.clear(); // 초기화
    std::vector<int> temp_indices; // 파싱 함수 전달용

    for (const auto& target : target_joints) {
        auto it = std::find(all_joint_names.begin(), all_joint_names.end(), target);
        if (it != all_joint_names.end()) {
            int index = std::distance(all_joint_names.begin(), it);
            target_indices_.push_back(index); // 멤버 변수 저장 (나중에 매핑할 때 사용)
            temp_indices.push_back(index);    // 파싱 함수 전달용
        }
    }

    parse_segmented_csv(theta_path, temp_indices, q_trajectory_);
    parse_segmented_csv(vel_path, temp_indices, q_dot_trajectory_);

    // Right arm elbow joint limit
    int raj4_local_idx = -1;
    for (size_t i = 0; i < target_joints.size(); ++i) {
        if (target_joints[i] == "RAJ4") {
            raj4_local_idx = i;
            break;
        }
    }

    if (raj4_local_idx != -1) {
        for (auto& q_vec : q_trajectory_) {
            if (q_vec(raj4_local_idx) > 1.3) {
                q_vec(raj4_local_idx) = 1.3;
            }
        }
    }

    if (!target_indices_.empty()) {
        joint_traj_.set_size(target_indices_.size()); 
    }

    robot_q_indices_ = TARGET_ROBOT_INDICES;
    if (robot_q_indices_.size() != target_indices_.size()) {
        std::cerr << "\033[31m" << "[Error] Mismatch between CSV columns and Robot Indices!" << "\033[0m" << std::endl;
    }
    
}

std::vector<std::string> CJointControl::read_csv_names(const std::string& path) {
    std::vector<std::string> names;
    std::ifstream file(path);
    std::string line;

    if (!file.is_open()) {
        std::cerr << "\033[31m" << "[Error] Cannot open file: " << path << "\033[0m" << std::endl;
        return names;
    }

    std::getline(file, line); 

    while (std::getline(file, line)) {
        line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());
        line.erase(std::remove(line.begin(), line.end(), '\n'), line.end());
        
        if(!line.empty() && line != "joint_names") { // 헤더가 "joint_names" 텍스트라면 제외
            names.push_back(line);
        }
    }
    file.close();
    return names;
}

void CJointControl::parse_segmented_csv(const std::string& path, 
                                        const std::vector<int>& target_indices, 
                                        std::vector<Eigen::VectorXd>& buffer) {
    std::ifstream file(path);
    std::string line;
    
    buffer.clear();
    int data_dim = target_indices.size();

    if (!file.is_open()) {
        std::cerr << "\033[31m" << "[Error] Cannot open file: " << path << "\033[0m" << std::endl;
        return;
    }

    std::getline(file, line); 

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<double> row_values;
        
        while (std::getline(ss, cell, ',')) {
            try {
                row_values.push_back(std::stod(cell));
            } catch (...) {
                row_values.push_back(0.0);
            }
        }

        if (!row_values.empty()) {
            Eigen::VectorXd vec_seg(data_dim);
            for (int i = 0; i < data_dim; ++i) {
                int original_col_idx = target_indices[i];
                
                // 범위 체크
                if (original_col_idx < row_values.size()) {
                    vec_seg(i) = row_values[original_col_idx];
                } else {
                    vec_seg(i) = 0.0; // 에러 방지용 0 처리
                }
            }
            buffer.push_back(vec_seg);
        }
    }
    file.close();
}

void CJointControl::ConvertStateMotor2Joint(uint8_t task, int nState, double dTime, double *dTheta, double *dThetaDot, double *dThetaAbs, double dNeckPitch, double dNeckYaw)
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

			m_bMotion = true;
			m_nMotionStep = 1;
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
			// m_bTorqueOff = true;
			m_bStandStill = true;

			m_bINIT = false;
			m_bHOMING = false;
			m_bCONTROL = false;
			m_bRL = false;
		}
		m_iPreviousControlMode = m_iControlMode;

		if (!m_bInitWrist) {
		//	(원본)
		m_vWristLinearAbs(0) =
			WristParallelMechanism
				.InverseKinematics_L(m_vMotorThetaAbsrad[L_wrist_roll_abs],
									m_vMotorThetaAbsrad[L_wrist_pitch_abs])
				.front_motor; // 5, 4
		m_vWristLinearAbs(1) =
			WristParallelMechanism
				.InverseKinematics_L(m_vMotorThetaAbsrad[L_wrist_roll_abs],
									m_vMotorThetaAbsrad[L_wrist_pitch_abs])
				.back_motor; // 5, 4

		m_vWristLinearAbs(2) =
			WristParallelMechanism
				.InverseKinematics_R(m_vMotorThetaAbsrad[R_wrist_roll_abs],
									m_vMotorThetaAbsrad[R_wrist_pitch_abs])
				.front_motor; // 12, 11
		m_vWristLinearAbs(3) =
			WristParallelMechanism
				.InverseKinematics_R(m_vMotorThetaAbsrad[R_wrist_roll_abs],
									m_vMotorThetaAbsrad[R_wrist_pitch_abs])
				.back_motor; // 12, 11

		
		//	(임시)
		m_vWristLinearAbs(2) =
			WristParallelMechanism
				.InverseKinematics_R(m_vMotorThetaAbsrad[R_wrist_roll_abs],
									m_vMotorThetaAbsrad[R_wrist_pitch_abs])
				.front_motor; // 12, 11
		m_vWristLinearAbs(3) =
			WristParallelMechanism
				.InverseKinematics_R(m_vMotorThetaAbsrad[R_wrist_roll_abs],
									m_vMotorThetaAbsrad[R_wrist_pitch_abs])
				.back_motor; // 12, 11

		m_bInitWrist = true;
		}
	}

	// for motion tracking
	m_nTaskNumPad = task;
	if((m_nTaskNumPad != m_nPreTaskNumPad) && m_iControlMode == CONTROL){
		m_nMotionIndex = m_nTaskNumPad;
		m_nMotionStep = 1;
		m_nNeckMotionStep = 1;
		m_dNeckControlStartTime = dTime;
	}
	m_nPreTaskNumPad = m_nTaskNumPad;

	m_nNeckMotionIndex = m_nTaskNumPad;
	m_dNeckPitchDes = dNeckPitch;
	m_dNeckYawDes = dNeckYaw;

	if(m_bTorqueOff){
			m_bINIT = false;
			m_bHOMING = false;
			m_bCONTROL = false;
			m_bRL = false;
	}

	//	(원본)
	dTheta[L_wrist_front_D] += m_vWristLinearAbs(0) * 1000;
	dTheta[L_wrist_back_D] += m_vWristLinearAbs(1) * 1000;

	

	//	(임시)
	dTheta[R_wrist_front_D] += m_vWristLinearAbs(2) * 1000;
	dTheta[R_wrist_back_D] += m_vWristLinearAbs(3) * 1000;

	// save motor data
    for(int i=0; i<m_nJDoF; i++)
    {
        m_vMotorThetarad(i) = dTheta[i];
        m_vMotorThetadotrad(i) = dThetaDot[i];
		m_vMotorThetaAbsrad(i) = dThetaAbs[i];
    }

	// cout<<m_vMotorThetarad[L_wrist_front_D]<<" , "<<m_vMotorThetarad[L_wrist_back_D]<<" , "<<m_vMotorThetarad[R_wrist_front_D]<<" , "<<m_vMotorThetarad[R_wrist_back_D]<<endl;
	// cout<<m_vMotorThetaAbsrad[L_wrist_front_D]<<" , "<<m_vMotorThetaAbsrad[L_wrist_back_D]<<" , "<<m_vMotorThetaAbsrad[R_wrist_front_D]<<" , "<<m_vMotorThetaAbsrad[R_wrist_back_D]<<endl<<endl<<endl;

	// save real time
	m_dTime = dTime;
	m_dInitTime = m_dTime - m_dInitStartTime;
	m_dHomingTime = m_dTime - m_dHomingStartTime;
	m_dControlTime = m_dTime - m_dControlStartTime;
	m_dRLTime = m_dTime - m_dRLStartTime;
	m_dNeckControlTime = m_dTime - m_dNeckControlStartTime;

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
		m_vPreMotorThetaradDes = m_vMotorThetaradDes;
		m_vPrMotorThetadotradDes = m_vMotorThetadotradDes;
	}

	// calculate Motor2Joint Jacobian
	m_mActuation_tau_q2psi = compute_tau_q2psi(m_vMotorThetarad);

	m_vJpsidot = m_mActuation_tau_q2psi * m_vMotorThetadotrad;

	// LPF qdot (Not used now) -> Please set _dt correctly in the low pass filter
	m_vJpsidot_filtered(0) = m_vJpsidot(0);
	m_vJpsidot_filtered(1) = m_vJpsidot(1);
	m_vJpsidot_filtered(2) = m_vJpsidot(2);
	m_vJpsidot_filtered(3) = m_vJpsidot(3);
	m_vJpsidot_filtered(4) = CustomMath::LowPassFilter(0.001, 300.0, m_vJpsidot(4), m_vJpsidot_filtered(4));
	m_vJpsidot_filtered(5) = CustomMath::LowPassFilter(0.001, 300.0, m_vJpsidot(5), m_vJpsidot_filtered(5));
	m_vJpsidot_filtered(6) = CustomMath::LowPassFilter(0.001, 300.0, m_vJpsidot(6), m_vJpsidot_filtered(6));

	m_vJpsidot_filtered(7) = m_vJpsidot(7);
	m_vJpsidot_filtered(8) = m_vJpsidot(8);
	m_vJpsidot_filtered(9) = m_vJpsidot(9);
	m_vJpsidot_filtered(10) = m_vJpsidot(10);
	m_vJpsidot_filtered(11) = CustomMath::LowPassFilter(0.001, 300.0, m_vJpsidot(11), m_vJpsidot_filtered(11));
	m_vJpsidot_filtered(12) = CustomMath::LowPassFilter(0.001, 300.0, m_vJpsidot(12), m_vJpsidot_filtered(12));
	m_vJpsidot_filtered(13) = CustomMath::LowPassFilter(0.001, 300.0, m_vJpsidot(13), m_vJpsidot_filtered(13));

	m_vJpsidot_filtered(14) = m_vJpsidot(14);
	m_vJpsidot_filtered(15) = m_vJpsidot(15);
	m_vJpsidot_filtered(16) = m_vJpsidot(16);
	m_vJpsidot_filtered(17) = m_vJpsidot(17);//CustomMath::LowPassFilter(0.001, 1000.0, m_vJpsidot(17), m_vJpsidot_filtered(17));
	m_vJpsidot_filtered(18) = m_vJpsidot(18);//CustomMath::LowPassFilter(0.001, 1000.0, m_vJpsidot(18), m_vJpsidot_filtered(18));

	m_vJpsidot_filtered.segment(19,31) = m_vJpsidot.segment(19,31);
	m_vJointQdotrad = m_vJpsidot_filtered;

	//	(원본)
	m_vJointQdotrad(L_wrist_roll) = (m_vJointQrad(L_wrist_roll) - m_vPreJointQrad(L_wrist_roll)) / 0.001;
	m_vJointQdotrad(L_wrist_pitch) = (m_vJointQrad(L_wrist_pitch) - m_vPreJointQrad(L_wrist_pitch)) / 0.001;
	m_vJointQdotrad(R_wrist_roll) = (m_vJointQrad(R_wrist_roll) - m_vPreJointQrad(R_wrist_roll)) / 0.001;
	m_vJointQdotrad(R_wrist_pitch) = (m_vJointQrad(R_wrist_pitch) - m_vPreJointQrad(R_wrist_pitch)) / 0.001;

	
	

	m_vPreJointQrad = m_vJointQrad;

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
	if(m_bINIT || m_bHOMING || m_bCONTROL || m_bTorqueOff || m_bRL || m_bStandStill){
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
			// if(abs(m_vMotorThetadotrad(i)) > m_vMotorThetadotradLimit(i)){
			// 	m_nCoutVelLimitNum ++;
			// 	if(m_nCoutVelLimitNum > 10){
			// 		m_bTorqueOff = true;
			// 	}
			// 	std::cout << "\033[1;34m"; 
			// 	std::cout << "=======================\n";
			// 	std::cout << "m_dTime   " << m_dTime << "\n";
			// 	std::cout << "i: " << i << "\n";
			// 	std::cout << "MOTOR VEL LIMIT:" << ", "<< m_vMotorThetadotrad(i) << ", " <<m_vMotorThetadotradLimit(i) << "\n";
			// 	std::cout << "\033[0m"; 
			// 	std::cout<<endl;
			// }
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

		// if(m_bHOMING || m_bCONTROL || m_bTorqueOff || m_bRL){
		// 	m_vTorque = m_mKpj * (m_vPreJointSafeDes - m_vJointQrad) + m_mKdj * (- m_vJointQdotrad);
		// 	m_vRLTorque = m_mKpj * (m_vPreRLJointSafeDes - m_vJointQrad) + m_mKdj * (- m_vJointQdotrad);
		// }
		// else{
			m_vTorque.setZero();
			m_vRLTorque.setZero();
		// }
	}
	else if(!m_bTorqueOff){
		m_vPreJointSafeDes = m_vJointQradDes;
		m_vPreRLJointSafeDes = m_vRLJointQradDes;
	}

	if(m_bStandStill){
		m_vTorque.setZero();
		m_vRLTorque.setZero();
	}

	// m_vTorque.setZero();
	
	if(m_bINIT || m_bHOMING || m_bCONTROL || m_bRL){
		for(int i=0;i<33;i++){
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
		for(int i=33;i<m_nJDoF;i++){
			dTarget[i] = m_vJointQradDes(i);
		}
	}
	if(m_bTorqueOff || m_bStandStill){
		for(int i=0;i<m_nJDoF;i++){
			dTarget[i] = 0.0;
		}
	}
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
	if(m_bINIT && m_bStandStill == false){
		// Set cubic desired q, qdot target for Init
		if (m_bmotormotion == true)
		{
			// cout<<"start homing! : "<<m_dHomingTime<<endl;
			m_dmotiontime = 3.0;

			m_vmotorgoal.setZero();
			//	(원본)
			// 팔 벌리기
			m_vmotorgoal.segment(0,m_nJDoF) << -0.0, 0.0, -0.0, +0.01, -0.0, 0.0, 0.0,
												0.0, -0.00, +.0, -0.01, +0.0, 0.0, 0.0,
												-0.0, 0.0, 0.0, 0.09, 0.0,
												0.0, 0.2, -0.0, +0.0, -0.0, 0.0, -0.6,
												-0.0, -0.2, -0.0, +0.0, -0.0, 0.0, +0.6; 
			
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
			if(abs(m_vMotorThetarad(i) - m_vmotorgoal(i))<0.0001){
				m_vMotorThetaradDes(i) = 0.0;
				m_vMotorThetadotradDes(i) = 0.0;
			}
		}
		if(MotorTrajectory.check_trajectory_complete() == 1){
			m_bINITCOMPLETE = true;
		}
		m_vJointQradDes = m_vJointQrad;
		m_vJointQradDesPre = m_vJointQradDes;
		m_vJointQdotradDes.setZero();

		m_vJointQradDes.tail(40).setZero();
	}
	else if(m_bHOMING && m_bStandStill == false){
		// Set cubic desired q, qdot target for homing
		if (m_bjointmotion == true)
		{
			// cout<<"start homing! : "<<m_dHomingTime<<endl;
			m_dmotiontime = 3.0;

			// m_vjointgoal.setZero();
			// // 자세 종류 테스트 시 사용
			// m_vjointgoal.segment(0,m_nJDoF) << -0.0, 0.0, -0.0, +0.0, -0.0, 0.0, 0.0,
			// 									0.0, -0.00, +.0, -0.0, +0.0, 0.0, 0.0,
			// 									-0.0, 0.0, 0.0, 0.0, 0.0,
			// 									-1.4, 0.2, -0.0, +0.0, -0.0, 0.0, 0.0,
			// 									1.4, -0.2, -0.0, +0.0, -0.0, 0.0, 0.0; 
			// // RL offseet 변경 (여기서 수정하면 됨 @@@@@)
			// m_vjointgoal.segment(0,m_nJDoF) <<	0.0, 0.0, -0.035, +0.38, -0.33, 0.0, 0.0,
			// 									-0.0, -0.0, +0.035, -0.38, +0.33, 0.0, 0.0,
			// 									0.0, 0.0, 0.0, 0.09, 0.0,
			// 									0.2, 0.2, 0.18, -0.35, 0.0, 0.0, 0.0,
			// 									-0.2, -0.2, -0.18, 0.35, 0.0, 0.0, 0.0;

			m_vjointgoal.segment(0,m_nJDoF) <<	0.03, 0.03, -0.09, +0.38, -0.33, 0.0, 0.0,
												-0.03, -0.03, +0.09, -0.38, +0.33, 0.0, 0.0,
												0.0, 0.0, 0.0, 0.09, 0.0,
												0.05, 0.2, 0.18, -0.55, 0.0, 0.0, 0.0,
												-0.05, -0.2, -0.18, 0.55, 0.0, 0.0, 0.0;								
			
			// PACE
			// // 0점s 자세

			//	(원본)
			// m_vjointgoal.segment(0,m_nJDoF) << -0.21815, 0.0, -0.61085, 0.95993, 0.17453, 0.0, -0.34905,
			// 								0.21815, 0.0, 0.61085, -0.95993, -0.17453, 0.0, 0.34905,
			// 								0.0, 0.0, 0.0, 0.09, 0.0,
			// 								-1.134464014, 0.701696021, -0.0, -0.741764932, 0.0, 0.008726646, 0.104719755,
			// 								+1.134464014, -0.701696021, -0.0, +0.741764932, -0.0, -0.008726646, -0.104719755; 
		
											
			m_vjointdotgoal.setZero();

			m_dmotionendtime = m_dHomingTime + m_dmotiontime-0.0005;
			JointTrajectory.reset_initial(m_dHomingTime, m_vJointQradDes, m_vJointQdotradDes);
			JointTrajectory.update_goal(m_vjointgoal, m_vjointdotgoal, m_dmotionendtime);

			// for motor tuning
			// m_vmotorgoal.setZero();
			// m_vmotorgoal.segment(0,m_nJDoF) << 

			// m_vmotordotgoal.setZero();

			// m_dmotionendtime = m_dHomingTime + m_dmotiontime-0.0005;
			// MotorTrajectory.reset_initial(m_dHomingTime, m_vMotorThetarad, m_vMotorThetadotrad);
			// MotorTrajectory.update_goal(m_vmotorgoal, m_vmotordotgoal, m_dmotionendtime);

			m_bjointmotion = false;
		}
		
		JointTrajectory.update_time(m_dHomingTime);
		m_vJointQradDes = JointTrajectory.position_cubicSpline();
		m_vJointQdotradDes = JointTrajectory.velocity_cubicSpline();

		for(int i = 0; i<m_nJDoF; i++){
			if(abs(m_vJointQrad(i) - m_vjointgoal(i))<0.0001){
				m_vJointQradDes(i) = m_vjointgoal(i);
				m_vJointQdotradDes(i) = 0.0;
			}
		}

		// MotorTrajectory.update_time(m_dHomingTime);
		// m_vMotorThetaradDes = MotorTrajectory.position_cubicSpline();
		// m_vMotorThetadotradDes = MotorTrajectory.velocity_cubicSpline();

		// for(int i = 0; i<m_nJDoF; i++){
		// 	if(abs(m_vMotorThetarad(i) - m_vmotorgoal(i))<0.001){
		// 		m_vMotorThetaradDes(i) = m_vmotorgoal(i);
		// 		m_vMotorThetadotradDes(i) = 0.0;
		// 	}
		// }

		m_vRLJointQradDes = m_vJointQradDes;
		m_vRLJointQradDesLPF_Pre = m_vJointQradDes;

		m_dPreNeckPitchDes = m_vJointQradDes(17);
		m_dPreNeckYawDes = m_vJointQradDes(18);
		m_dPreNeckPitchDotDes = m_vJointQdotradDes(17);
		m_dPreNeckYawDotDes   = m_vJointQdotradDes(18);

		m_vPreNeckgoal << m_dPreNeckPitchDes, m_dPreNeckYawDes;
		m_vPreNeckdotgoal.setZero();

		m_vPreWaistgoal << m_vJointQradDes(14), m_vJointQradDes(15), m_vJointQradDes(16);
		m_vPreWaistdotgoal.setZero();
	}

	else if(m_bCONTROL && m_bStandStill == false){
	// Set sin desired q, qdot targetv
	// m_bMotion = false;
		if(!m_bMotion){
			// PACE
			double f = 0.1;   // sin 주파수(Hz)
			double f0 = 0.001;   // 시작 주파수(Hz)
			double f1 = 0.8;  // 끝 주파수(Hz)
			double T  = 10;  // 총 시간

			double t = m_dControlTime;  // 0~20
			if(t>=T){
				t = T;
			}
			// double phi = 2 * M_PI * (f0 * t + (f1 - f0) * t * t / (2 * T));
			double phi = 2 * M_PI * f * t;

			m_vJointQradDes.segment(0,m_nJDoF) <<
			
				// PACE
				//	(원본)
				-0.56725*0.0*pow(sin(phi / 2.0), 2),
				0.3491*0.0*sin(phi), 
				1.13445*0.0*sin(phi), 
				0.95993*0.0*sin(phi), 
				0.69813*0.0*sin(phi), 
				-0.3491*0.0*sin(phi),
				0.69815*0.0*sin(phi),

				0.56725*0.0*pow(sin(phi / 2.0), 2),
				-0.3491*0.0*sin(phi), 
				1.13445*0.0*sin(phi), 
				0.95993*0.0*sin(phi), 
				0.69813*0.0*sin(phi), 
				0.3491*0.0*sin(phi), 
				0.69815*0.0*sin(phi),

				0.523599*0.0*sin(phi),
				0.523599*0.0*sin(phi),
				1.5708*0.0*pow(sin(phi / 2.0), 2),
				-0.7854*0.0*pow(sin(phi / 2.0), 2), 
				1.2217*0.0*sin(phi),

				2.00712864*0.0*sin(phi),
				0.5123*0.0*sin(phi),
				-1.658062789*0.0*pow(sin(phi / 2.0), 2),
				0.916297857*0.0*sin(phi),
				-1.134464014*0.0*sin(phi),
				-0.514872129*0.5*sin(phi),
				0.977384381*0.0*sin(phi),

				2.00712864*0.0*sin(phi),
				-0.5123*0.0*sin(phi), 
				1.658062789*0.0*pow(sin(phi / 2.0), 2),
				0.916297857*0.0*sin(phi), 
				-1.134464014*0.0*sin(phi),         
				0.514872129*0.5*sin(phi), 
				0.977384381*0.0*sin(phi);

				// 유선 모션 트라젝토리
				// if (playback_tick_ < q_interp_buffer_.size()) {
				//     Eigen::VectorXd q_segment_des = q_interp_buffer_[playback_tick_];
				//     Eigen::VectorXd q_dot_segment_des = q_dot_interp_buffer_[playback_tick_];

				//     for(size_t i = 0; i < robot_q_indices_.size(); ++i) {
				//         int robot_idx = robot_q_indices_[i];
						
				//         // CSV의 i번째 데이터를 -> 로봇의 robot_idx번째 관절에 할당
				//         m_vJointQradDes(robot_idx) 		= q_segment_des(i);
				//         m_vJointQdotradDes(robot_idx) 	= q_dot_segment_des(i);
				//     }
					
				//     playback_tick_++;

				// } else {
				//     if (!q_interp_buffer_.empty()) {
				//         Eigen::VectorXd q_final = q_interp_buffer_.back();
				//         for(size_t i = 0; i < robot_q_indices_.size(); ++i) {
				//             int robot_idx = robot_q_indices_[i];
				//             m_vJointQradDes(robot_idx) = q_final(i);
				//             m_vJointQdotradDes(robot_idx) = 0.0; 
				//         }
				//     }
				// }

				// m_vJointQdotradDes.setZero();
				// cout << "m_vJointQradDes : " << m_vJointQradDes.transpose() << endl;

			m_vJointQradDes += m_vPreJointQradDes;
			m_vJointQdotradDes.setZero();

			if (m_bjointmotion == true && t>=T)
			{
				m_dmotiontime = 5.0;

				// m_vjointgoal.setZero();
				// // 자세 종류 테스트 시 사용
				m_vjointgoal.segment(0,m_nJDoF) << -0.0, 0.0, -0.0, +0.01, -0.0, 0.0, 0.0,
													0.0, -0.00, +.0, -0.01, +0.0, 0.0, 0.0,
													-0.0, 0.0, 0.0, 0.09, 0.0,
													0.0, 0.2, -0.0, +0.0, -0.0, 0.0, 0.0,
													-0.0, -0.2, -0.0, +0.0, -0.0, 0.0, 0.0; //m_vhomeposition
				// // RL offseet 변경 (여기서 수정하면 됨 @@@@@)
				// m_vjointgoal.segment(0,m_nJDoF) <<	0.03, 0.06, -0.07, +0.38, -0.33, 0.0, 0.0,
				// 									-0.03, -0.06, +0.07, -0.38, +0.33, 0.0, 0.0,
				// 									0.0, 0.0, 0.0;
				// PACE
				// // 0점s 자세
				//	(원본)
				// m_vjointgoal.segment(0,m_nJDoF) << 0.0, 0.0, -0.0, 0.1, -0.0, 0.0, 0.0,
				// 								0.0, 0.0, 0.0, -0.1, 0.0, 0.0, 0.0,
				// 								0.0, 0.0, 0.0, 0.2, 0.0,
				// 								0.0, 0.2, -0.0, +0.0, -0.0, 0.0, 0.0,
				// 								-0.0, -0.2, -0.0, +0.0, -0.0, 0.0, 0.0; 

				// m_vjointgoal.segment(0,m_nJDoF) <<	0.0, 0.0, -0.035, +0.38, -0.33, 0.0, 0.0,
				// 									-0.0, -0.0, +0.035, -0.38, +0.33, 0.0, 0.0,
				// 									0.0, 0.0, 0.2, 0.0, 0.0,
				// 									0.0, 0.5123, 0.18, -0.35, 0.0, 0.0, 0.0,
				// 									-0.0, -0.5123, -0.18, 0.35, 0.0, 0.0, 0.0;

		
				
				
				m_vjointdotgoal.setZero();

				m_dmotionendtime = m_dControlTime + m_dmotiontime-0.0005;
				JointTrajectory.reset_initial(m_dControlTime, m_vJointQradDes, m_vJointQdotradDes);
				JointTrajectory.update_goal(m_vjointgoal, m_vjointdotgoal, m_dmotionendtime);

				m_bjointmotion = false;
			}
			
			if(t>=T){
				JointTrajectory.update_time(m_dControlTime);
				m_vJointQradDes = JointTrajectory.position_cubicSpline();
				m_vJointQdotradDes = JointTrajectory.velocity_cubicSpline();

				for(int i = 0; i<m_nJDoF; i++){
					if(abs(m_vJointQrad(i) - m_vjointgoal(i))<0.001){
						m_vJointQradDes(i) = m_vjointgoal(i);
						m_vJointQdotradDes(i) = 0.0;
					}
				}
			}

			// 재완
			// Set sin desired psi, psidot targetv
			// m_vMotorThetaradDes.segment(0,m_nJDoF) <<
			// 	-0.0*M_PI/180.0*sin(1.5*(m_dControlTime)),//*sin(2*(m_dControlTime)),
			// 	0.0*M_PI/180.0*sin(1.5*(m_dControlTime)), 
			// 	-10.0*M_PI/180.0*sin(1.0*(m_dControlTime)), //*sin(1.0*(m_dControlTime)), 
			// 	90.0*M_PI/180.0*sin(1.5*(m_dControlTime))*sin(1.5*(m_dControlTime)), 
			// 	40.0*M_PI/180.0*sin(1.0*(m_dControlTime)), 
			// 	40.0*M_PI/180.0*sin(1.0*(m_dControlTime)),
			// 	25.0*M_PI/180.0*sin(1.0*(m_dControlTime)),

			// 	0*M_PI/180.0*sin(1.5*(m_dControlTime)),//*sin(2*(m_dControlTime)),
			// 	-0.0*M_PI/180.0*sin(1.5*(m_dControlTime)), 
			// 	50.0*M_PI/180.0*sin(1.0*(m_dControlTime)), //*sin(1.0*(m_dControlTime)), 
			// 	-90.0*M_PI/180.0*sin(1.5*(m_dControlTime))*sin(1.5*(m_dControlTime)), 
			// 	-40.0*M_PI/180.0*sin(1.0*(m_dControlTime)), 
			// 	40.0*M_PI/180.0*sin(1.0*(m_dControlTime)), 
			// 	-25.0*M_PI/180.0*sin(1.0*(m_dControlTime)),

			// 	0.0*sin(1.0*(m_dControlTime)),
			// 	0.0*sin(1.0*(m_dControlTime)),
			// 	0.0*sin(1.0*(m_dControlTime));

			// m_vMotorThetaradDes += m_vPreMotorThetaradDes;
			// m_vMotorThetadotradDes.setZero();
		}
		if(m_bMotion){	
			// 상체 모션 트래킹제어
			// 1. 현재 스테이트 확인 후 1~8번 모션 선택 or idle 모션 선택(1~8가 아닌 경우, 0번으로 칭함) 선택한 모션으로 조인트 트라젝토리 생성

			// 2. 생성된 트라젝토리로 이동

			// 3. 선택된 모션 0~8 50hz->1khz 보간 제어 및 lpf 필터링

			// 4. 만약 스테이트가 9가 들어오면 현재 디자이얼에서 홈포지션으로 조인트 트라젝토리 생성 -> 홈포지션으로 트라젝토리 이동 후 유지

			// 5. 수행 중인 동작이 끝나면 0번 동작 반복

			if(m_nMotionIndex == 9){
				if (m_nMotionStep == 1) {
					m_vjointgoal.segment(0,m_nJDoF) = m_vhomeposition.head(m_nJDoF);
					m_vjointdotgoal.setZero();

					m_dmotiontime = 2.0;

					m_dmotionendtime = m_dControlTime + m_dmotiontime-0.0005;
					JointTrajectory.reset_initial(m_dControlTime, m_vJointQradDes, m_vJointQdotradDes);
					JointTrajectory.update_goal(m_vjointgoal, m_vjointdotgoal, m_dmotionendtime);
					
					std::cout << ">>> [Transition] Moving to Motion Home Pose..." << std::endl;
					m_nMotionStep = 2; 
				}
				if (m_nMotionStep == 2) {
					JointTrajectory.update_time(m_dControlTime);
					m_vJointQradDes = JointTrajectory.position_cubicSpline();
					m_vJointQdotradDes = JointTrajectory.velocity_cubicSpline();

					for(int i = 0; i < NUM_OF_JOINTS; i++){
						if(abs(m_vjointgoal(i) - m_vJointQrad(i))<0.001){
							m_vJointQradDes(i) = m_vjointgoal(i);
							m_vJointQdotradDes(i) = 0.0;
						}
					}
					
				}
			}
			else if(m_nMotionIndex != 9){
				if (m_nMotionStep == 1) {
					switch (m_nMotionIndex) {
						case 0: m_pCurrentMotionData = &m_MotionData0; cout<<"motion 0"<<endl; break;
						case 1: m_pCurrentMotionData = &m_MotionData1; cout<<"motion 1"<<endl; break;
						case 2: m_pCurrentMotionData = &m_MotionData2; cout<<"motion 2"<<endl; break;
						case 3: m_pCurrentMotionData = &m_MotionData3; cout<<"motion 3"<<endl; break;
						case 4: m_pCurrentMotionData = &m_MotionData4; cout<<"motion 4"<<endl; break;
						case 5: m_pCurrentMotionData = &m_MotionData5; cout<<"motion 5"<<endl; break;
						case 6: m_pCurrentMotionData = &m_MotionData6; cout<<"motion 6"<<endl; break;
						case 7: m_pCurrentMotionData = &m_MotionData7; cout<<"motion 7"<<endl; break;
						case 8: m_pCurrentMotionData = &m_MotionData8; cout<<"motion 8"<<endl; break;
						case 999: m_pCurrentMotionData = &m_MotionData0; cout<<"motion 0"<<endl; break;
						default: m_pCurrentMotionData = m_pCurrentMotionData; cout<<"same motion"<<endl; break; // 1~8가 아니면 0번(Idle)
					}

					m_vMotionStartQ = (*m_pCurrentMotionData)[0];
					m_vMotionStartQdot.setZero();

					m_dTransitionTime = 0.5; // 2.0; // interpolation time

					m_dmotionendtime = m_dControlTime + m_dTransitionTime;
					
					JointTrajectory.reset_initial(m_dControlTime, m_vJointQradDes, m_vJointQdotradDes);
					JointTrajectory.update_goal(m_vMotionStartQ, m_vMotionStartQdot, m_dmotionendtime);

					if(m_nMotionIndex == 0  || m_nMotionIndex == 999){
						cout << ">>> [Transition] Moving to Motion Idle Pose..." << endl;
					}
					else{
						cout << ">>> [Transition] Moving to Motion Pose..." << endl;
					}
					m_nMotionStep = 2; 

					if(m_nMotionIndex == 6){						//악수
						m_nEndHandshake = 0;
					}
				}
				if (m_nMotionStep == 2) {
					JointTrajectory.update_time(m_dControlTime);
					m_vJointQradDes = JointTrajectory.position_cubicSpline();
					m_vJointQdotradDes = JointTrajectory.velocity_cubicSpline();

					for(int i = 0; i < NUM_OF_JOINTS; i++){
						if(abs(m_vMotionStartQ(i) - m_vJointQrad(i))<0.001){
							m_vJointQradDes(i) = m_vMotionStartQ(i);
							m_vJointQdotradDes(i) = 0.0;
						}
					}

					if (m_dControlTime >= m_dmotionendtime) {
						m_dMotionStartTime = m_dControlTime;
						std::cout << ">>> [Playback] Motion Start!" << std::endl;
						m_nMotionStep = 3; 
						m_vJointQradDesPre = m_vJointQradDes;
						m_vPreJointQdotradDes = m_vJointQdotradDes;
					}
				}
				if (m_nMotionStep == 3) {
					m_dMotionplaybackspeed = 1.0;	// => 속도 조절 가능

					if(m_nMotionIndex == 1){
						m_dMotionplaybackspeed = 1.0;
					}
					else if(m_nMotionIndex == 7){
						m_dMotionplaybackspeed = 0.75;
					}

					double dElapsedTime = (m_dControlTime - m_dMotionStartTime) * m_dMotionplaybackspeed;
					double dCsvDt = 0.0333;	// csv 저장 hz 

					double dCurrentIndex = dElapsedTime / dCsvDt;
					int nIdxA = (int)dCurrentIndex;
					m_nMotionIndexNum = nIdxA;
					int nIdxB = nIdxA + 1;
					double alpha = dCurrentIndex - nIdxA; 
					if (nIdxB < m_pCurrentMotionData->size()) {
						m_vqA = (*m_pCurrentMotionData)[nIdxA];
						m_vqB = (*m_pCurrentMotionData)[nIdxB];
						m_vJointQradDes = (1.0 - alpha) * m_vqA + alpha * m_vqB;
						m_vJointQdotradDes = (m_vqB-m_vqA)/dCsvDt;
						for(int i = 0; i < m_nJDoF; i++) {
							m_vJointQradDes(i) = CustomMath::LowPassFilter(0.001, 100.0 * 2.0 * 3.141592, m_vJointQradDes(i), m_vJointQradDesPre(i));
							m_vJointQdotradDes(i) = CustomMath::LowPassFilter(0.001, 10.0 * 2.0 * 3.141592, m_vJointQdotradDes(i), m_vPreJointQdotradDes(i));
							m_vJointQradDesPre(i) = m_vJointQradDes(i);
							m_vPreJointQdotradDes(i) = m_vJointQdotradDes(i);
						}
						// m_vJointQdotradDes.setZero();
					} 
					else {
						m_vJointQradDes = m_pCurrentMotionData->back();
						m_vJointQdotradDes.setZero();
						if(m_nMotionIndex == 0 || m_nMotionIndex == 999){	//idle 모션일 때
							m_dMotionStartTime = m_dControlTime;
							m_nMotionIndex = 0;
							// m_nMotionStep = 1;
						}
						else if(m_nMotionIndex == 1){						//서서인사중
							// m_dMotionStartTime = m_dControlTime;
							m_nMotionIndex = m_nMotionIndex;
							m_nMotionStep = 1;
						}
						else if(m_nMotionIndex == 6){						//악수
							m_nMotionIndex = m_nMotionIndex;
							// m_nEndHandshake = 0;
						}
						else if(m_nMotionIndex == 7 || m_nMotionIndex == 8){//사진촬영
							m_nMotionIndex = m_nMotionIndex;
						}
						else{
							std::cout << ">>> [Playback] Finished. Returning to Idle." << std::endl;
							m_nMotionIndex = 0;
							m_nMotionStep = 1;
						}
					}
				}

				// 1. Body Joints (0~13) Fixed
				m_vJointQradDes.head(14) = m_vhomeposition.head(14);
				m_vJointQdotradDes.head(14).setZero();		
				
				// 롤
				m_vJointQradDes(14) = 0.0;	
				m_vJointQdotradDes(14) = 0.0;

				// 요
				// m_vJointQradDes(15) = 0.0;	
				// m_vJointQdotradDes(15) = 0.0;

				// 피치
				// m_vJointQradDes(16) = 0.0;	
				// m_vJointQdotradDes(16) = 0.0;		

				// 3. Waist & Neck Control (17, 18) 
				if(m_nNeckMotionStep == 1 && (m_nNeckMotionIndex == 0 || m_nNeckMotionIndex == 1 || m_nNeckMotionIndex == 7 || m_nNeckMotionIndex == 8)){
					
					m_bneckmotionfirst = true;
					if (m_nNeckMotionIndex == 0 || m_nNeckMotionIndex == 7 || m_nNeckMotionIndex == 8) {			
						m_vNeckgoal(0) = m_vhomeposition(17);
						m_vNeckgoal(1) = m_vhomeposition(18);
						m_vNeckdotgoal.setZero();

						m_vWaistgoal(0) = m_vhomeposition(14); 		//r
						m_vWaistgoal(1) = m_vhomeposition(15); 		//y
						m_vWaistgoal(2) = m_vhomeposition(16); 		//p
						m_vWaistdotgoal.setZero();
					}
					else if (m_nNeckMotionIndex == 1){
						m_vNeckgoal(0) = m_vhomeposition(17);
						m_vNeckgoal(1) = 0.5 * 1.2217 * sin(m_dControlTime * 2 * 3.141592);			// 랜덤 넥 요
						m_vNeckdotgoal.setZero();

						m_vWaistgoal(0) = m_vhomeposition(14); 		//r
						m_vWaistgoal(1) = m_vNeckgoal(1) * 0.5;		//y
						m_vWaistgoal(2) = m_vhomeposition(16); 		//p
						m_vWaistdotgoal.setZero();
					}

					m_dneckmotiontime = 1.0;
				
					m_dneckmotionendtime = m_dNeckControlTime + m_dneckmotiontime-0.0005;
					NeckTrajectory.reset_initial(m_dNeckControlTime, m_vPreNeckgoal, m_vPreNeckdotgoal);
					NeckTrajectory.update_goal(m_vNeckgoal, m_vNeckdotgoal, m_dneckmotionendtime);

					m_dneckmotionendtime = m_dNeckControlTime + m_dneckmotiontime-0.0005;
					WaistTrajectory.reset_initial(m_dNeckControlTime, m_vPreWaistgoal, m_vPreWaistdotgoal);
					WaistTrajectory.update_goal(m_vWaistgoal, m_vWaistdotgoal, m_dneckmotionendtime);
					
					std::cout << ">>> [Transition] Neck Moving to Motion Home Pose..."<<0.5 * 1.2217 * sin(m_dControlTime * 2 * 3.141592)<< std::endl;
					m_nNeckMotionStep = 2; 
					m_nNeckFilter = 0.001;
					
				}
				if (m_nNeckMotionStep == 2 && (m_nNeckMotionIndex == 0 || m_nNeckMotionIndex == 1 || m_nNeckMotionIndex == 7 || m_nNeckMotionIndex == 8)){
					NeckTrajectory.update_time(m_dNeckControlTime);
					m_vJointQradDes(17) = NeckTrajectory.position_cubicSpline()(0);
					m_vJointQdotradDes(17) = NeckTrajectory.velocity_cubicSpline()(0);
					m_vJointQradDes(18) = NeckTrajectory.position_cubicSpline()(1);
					m_vJointQdotradDes(18) = NeckTrajectory.velocity_cubicSpline()(1);
					

					if(abs(m_vNeckgoal(0) - m_vJointQradDes(17))<0.001){
						m_vJointQradDes(17) = m_vNeckgoal(0);
						m_vJointQdotradDes(17) = 0.0;
					}
					if(abs(m_vNeckgoal(1) - m_vJointQradDes(18))<0.001){
						m_vJointQradDes(18) = m_vNeckgoal(1);
						m_vJointQdotradDes(18) = 0.0;
					}				
					

					WaistTrajectory.update_time(m_dNeckControlTime);
					m_vJointQradDes(14) = WaistTrajectory.position_cubicSpline()(0);
					m_vJointQdotradDes(14) = WaistTrajectory.velocity_cubicSpline()(0);
					m_vJointQradDes(15) = WaistTrajectory.position_cubicSpline()(1);
					m_vJointQdotradDes(15) = WaistTrajectory.velocity_cubicSpline()(1);
					m_vJointQradDes(16) = WaistTrajectory.position_cubicSpline()(2);
					m_vJointQdotradDes(16) = WaistTrajectory.velocity_cubicSpline()(2);


					if(abs(m_vWaistgoal(0) - m_vJointQradDes(14))<0.001){
						m_vJointQradDes(14) = m_vWaistgoal(0);
						m_vJointQdotradDes(14) = 0.0;
					}		
					if(abs(m_vWaistgoal(1) - m_vJointQradDes(15))<0.001){
						m_vJointQradDes(15) = m_vWaistgoal(1);
						m_vJointQdotradDes(15) = 0.0;
					}	
					if(abs(m_vWaistgoal(2) - m_vJointQradDes(16))<0.001){
						m_vJointQradDes(16) = m_vWaistgoal(2);
						m_vJointQdotradDes(16) = 0.0;
					}		
					

					if (m_dNeckControlTime >= m_dneckmotionendtime) {
						std::cout << ">>> [Playback] Neck Motion Start!" << std::endl;
						m_nNeckMotionStep = 3; 
					}
					m_dPreNeckPitchDes = m_vJointQradDes(17);
					m_dPreNeckYawDes = m_vJointQradDes(18);
					m_dPreNeckPitchDotDes = m_vJointQdotradDes(17);
					m_dPreNeckYawDotDes   = m_vJointQdotradDes(18);

					m_vPreNeckgoal << m_vJointQradDes(17), m_vJointQradDes(18);
					m_vPreNeckdotgoal << m_vJointQdotradDes(17), m_vJointQdotradDes(18);
					m_vPreWaistgoal << m_vJointQradDes(14), m_vJointQradDes(15), m_vJointQradDes(16);
					m_vPreWaistdotgoal << m_vJointQdotradDes(14), m_vJointQdotradDes(15), m_vJointQdotradDes(16);
					
				}
				if (m_nNeckMotionStep == 3 && (m_nNeckMotionIndex == 0 || m_nNeckMotionIndex == 1 || m_nNeckMotionIndex == 7 || m_nNeckMotionIndex == 8)){
					
					if (m_nNeckMotionIndex == 0  || m_nNeckMotionIndex == 7 || m_nNeckMotionIndex == 8) {
						m_vJointQradDes.head(14) = m_vhomeposition.head(14);								
						m_vJointQdotradDes.head(14).setZero();

						m_vJointQradDes(17) = m_vPreNeckgoal(0);
						m_vJointQdotradDes(17) = m_vPreNeckdotgoal(0);
						m_vJointQradDes(18) = m_vPreNeckgoal(1);
						m_vJointQdotradDes(18) = m_vPreNeckdotgoal(1);

						m_vJointQradDes(14) = m_vPreWaistgoal(0);
						m_vJointQdotradDes(14) = m_vPreWaistdotgoal(0);
						m_vJointQradDes(15) = m_vPreWaistgoal(1);
						m_vJointQdotradDes(15) = m_vPreWaistdotgoal(1);
						m_vJointQradDes(16) = m_vPreWaistgoal(2);
						m_vJointQdotradDes(16) = m_vPreWaistdotgoal(2);
					}
					if (m_nNeckMotionIndex == 1) {						
						m_vJointQradDes.head(14) = m_vhomeposition.head(14);								
						m_vJointQdotradDes.head(14).setZero();

						m_vJointQradDes(17) = m_vPreNeckgoal(0);
						m_vJointQdotradDes(17) = m_vPreNeckdotgoal(0);
						m_vJointQradDes(18) = m_vPreNeckgoal(1);
						m_vJointQdotradDes(18) = m_vPreNeckdotgoal(1);

						m_vJointQradDes(14) = m_vPreWaistgoal(0);
						m_vJointQdotradDes(14) = m_vPreWaistdotgoal(0);
						m_vJointQradDes(15) = m_vPreWaistgoal(1);
						m_vJointQdotradDes(15) = m_vPreWaistdotgoal(1);
						m_vJointQradDes(16) = m_vPreWaistgoal(2);
						m_vJointQdotradDes(16) = m_vPreWaistdotgoal(2);

						if(m_dRandomNeckWaistcount >= std::pow(sin(m_dControlTime * 2 * 3.141592),2)*1000 + 3300){
							m_nNeckMotionStep = 1;
							m_dRandomNeckWaistcount = 0.0;

							cout<<">>> [Transition] Neck Random Motion Change Pose..."<<m_dRandomNeckWaistcount<<endl;
						}
						m_dRandomNeckWaistcount += 1.0;
					}	

					m_vPreNeckgoal << m_vJointQradDes(17), m_vJointQradDes(18);
					m_vPreNeckdotgoal << m_vJointQdotradDes(17), m_vJointQdotradDes(18);
					m_vPreWaistgoal << m_vJointQradDes(14), m_vJointQradDes(15), m_vJointQradDes(16);
					m_vPreWaistdotgoal << m_vJointQdotradDes(14), m_vJointQdotradDes(15), m_vJointQdotradDes(16);
					
				}

				if ((m_nNeckMotionIndex != 0) && (m_nNeckMotionIndex != 1) && (m_nNeckMotionIndex != 7) && (m_nNeckMotionIndex != 8)) {
					m_vJointQradDes(14) = 0.0; 			//롤
					// m_vJointQradDes(15) = 0.0; 		//요
					// m_vJointQradDes(16) = 0.0; 			//피치
					
					m_vJointQdotradDes(14) = 0.0;		//롤
					// m_vJointQdotradDes(15) = 0.0;	//요
					// m_vJointQdotradDes(16) = 0.0;		//피치

					const double dInputDt   = 0.4;      // 보간 주기

					// 3. 큐빅 보간 (보간 시간 가변)
					constexpr double eps = 1e-9;
					static double s_prevPitchIn = 0.0;
					static double s_prevYawIn   = 0.0;
					static bool   s_prevInit    = false;

					// 입력값
					const double pitchIn = m_dNeckPitchDes;
					const double yawIn   = m_dNeckYawDes;


					bool bChanged = false;

					if (!s_prevInit) {
						s_prevPitchIn = pitchIn;
						s_prevYawIn   = yawIn;
						s_prevInit    = true;
						bChanged      = true;   // 첫 진입 시 1회 세팅
					} else {
						bChanged =
							(std::abs(pitchIn - s_prevPitchIn) > eps) ||
							(std::abs(yawIn   - s_prevYawIn)   > eps);
					}

					// if ((bChanged && m_nNeckMotioncount>=dInputDt*1000) || m_bneckmotionfirst)
					if ((bChanged) || m_bneckmotionfirst)
					{
						// 첫 진입 처리(요청대로 유지)
						if (m_bneckmotionfirst) {
							m_nNeckMotioncount = 0;
							m_bneckmotionfirst = false;
						}

						// CMD1 갱신: "현재 입력"을 목표로 저장
						m_dPitchCmd1 = pitchIn;
						m_dYawCmd1   = yawIn;

						// 스플라인 시간 구간(가변 튜닝)
						const double t0 = m_dControlTime;
						const double tf = t0 + dInputDt;

						// 목표 위치: CMD1
						const Eigen::Vector2d v_goal(m_dPitchCmd1, m_dYawCmd1);

						// 목표 속도: 0
						const Eigen::Vector2d v_dotgoal(0.0, 0.0);

						// 시작 상태: 현재 출력(직전 필터 상태)
						const Eigen::Vector2d v_start(m_dPreNeckPitchDes,     m_dPreNeckYawDes);
						const Eigen::Vector2d vdot_start(m_dPreNeckPitchDotDes, m_dPreNeckYawDotDes);

						NeckTrackingTrajectory.reset_initial(t0, v_start, vdot_start);
						NeckTrackingTrajectory.update_goal(v_goal, v_dotgoal, tf);

						// 이전 입력 갱신
						s_prevPitchIn = pitchIn;
						s_prevYawIn   = yawIn;

						m_nNeckMotioncount = 0; // (원 코드 유지)
					}
					m_nNeckMotioncount++;

					NeckTrackingTrajectory.update_time(m_dControlTime);

					double pitchCmd    = NeckTrackingTrajectory.position_cubicSpline()(0);
					double yawCmd      = NeckTrackingTrajectory.position_cubicSpline()(1);
					
					double pitchDotCmd = NeckTrackingTrajectory.velocity_cubicSpline()(0);
					double yawDotCmd   = NeckTrackingTrajectory.velocity_cubicSpline()(1);

					// 컷오프 주파수 증가
					m_nNeckFilter += 0.001;
					// control dControlDt
					if (m_nNeckFilter > 2.0) {
						m_nNeckFilter = 2.0;
					}

					// Low-pass filter
					pitchCmd = CustomMath::LowPassFilter( 0.001, m_nNeckFilter * 500 * 2 * 3.141592, pitchCmd, m_dPreNeckPitchDes);
					yawCmd = CustomMath::LowPassFilter( 0.001, m_nNeckFilter * 500 * 2 * 3.141592, yawCmd, m_dPreNeckYawDes);
					pitchDotCmd = CustomMath::LowPassFilter( 0.001, m_nNeckFilter * 100 * 2 * 3.141592, pitchDotCmd, m_dPreNeckPitchDotDes);
					yawDotCmd = CustomMath::LowPassFilter( 0.001, m_nNeckFilter * 100 * 2 * 3.141592, yawDotCmd, m_dPreNeckYawDotDes);

					m_vJointQradDes(17) = pitchCmd;
					m_vJointQradDes(18) = yawCmd;

					m_vJointQdotradDes(17) = pitchDotCmd;
					m_vJointQdotradDes(18) = yawDotCmd;

					m_dPreNeckPitchDes = m_vJointQradDes(17);
					m_dPreNeckYawDes   = m_vJointQradDes(18);

					m_dPreNeckPitchDotDes = m_vJointQdotradDes(17);
					m_dPreNeckYawDotDes   = m_vJointQdotradDes(18);

					m_vPreNeckgoal << m_vJointQradDes(17), m_vJointQradDes(18);
					m_vPreNeckdotgoal << m_vJointQdotradDes(17), m_vJointQdotradDes(18);
					m_vPreWaistgoal << m_vJointQradDes(14), m_vJointQradDes(15), m_vJointQradDes(16);
					m_vPreWaistdotgoal << m_vJointQdotradDes(14), m_vJointQdotradDes(15), m_vJointQdotradDes(16);
				}
			}
		}
	}

	if(m_bStandStill){
		m_vJointQradDes = m_vJointQradDes;
		m_vJointQdotradDes.setZero();
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

	m_vTorque.setZero();
	m_vRLTorque.setZero();

	// Joint Task
	if(m_bINIT){
		m_vTorque = (m_mKpj*(m_vMotorThetaradDes - m_vMotorThetarad) + m_mKdj*(m_vMotorThetadotradDes - m_vMotorThetadotrad));
		
		//	(원본)
		m_vTorque(L_wrist_front_D) = (0.030123*(m_vMotorThetaradDes(L_wrist_front_D) - m_vMotorThetarad(L_wrist_front_D)) + 0.000153*(m_vMotorThetadotradDes(L_wrist_front_D) - m_vMotorThetadotrad(L_wrist_front_D)));
		m_vTorque(L_wrist_back_D) = (0.030123*(m_vMotorThetaradDes(L_wrist_back_D) - m_vMotorThetarad(L_wrist_back_D)) + 0.000153*(m_vMotorThetadotradDes(L_wrist_back_D) - m_vMotorThetadotrad(L_wrist_back_D)));
		m_vTorque(L_wrist_yaw_D) = (m_mKpj(L_wrist_yaw,L_wrist_yaw)*(m_vMotorThetaradDes(L_wrist_yaw_D) - m_vMotorThetarad(L_wrist_yaw_D)) + m_mKdj(L_wrist_yaw, L_wrist_yaw)*(m_vMotorThetadotradDes(L_wrist_yaw_D) - m_vMotorThetadotrad(L_wrist_yaw_D)));
		
		m_vTorque(R_wrist_front_D) = (0.030123*(m_vMotorThetaradDes(R_wrist_front_D) - m_vMotorThetarad(R_wrist_front_D)) + 0.000153*(m_vMotorThetadotradDes(R_wrist_front_D) - m_vMotorThetadotrad(R_wrist_front_D)));
		m_vTorque(R_wrist_back_D) = (0.030123*(m_vMotorThetaradDes(R_wrist_back_D) - m_vMotorThetarad(R_wrist_back_D)) + 0.000153*(m_vMotorThetadotradDes(R_wrist_back_D) - m_vMotorThetadotrad(R_wrist_back_D)));
		m_vTorque(R_wrist_yaw_D) = (m_mKpj(R_wrist_yaw,R_wrist_yaw)*(m_vMotorThetaradDes(R_wrist_yaw_D) - m_vMotorThetarad(R_wrist_yaw_D)) + m_mKdj(R_wrist_yaw, R_wrist_yaw)*(m_vMotorThetadotradDes(R_wrist_yaw_D) - m_vMotorThetadotrad(R_wrist_yaw_D)));


		if(m_bINITCOMPLETE){
			// m_vTorque.setZero();
		}
	}
	else if(m_bHOMING || m_bCONTROL || m_bRL){
					
		m_vTorque = m_mActuation_tau_q2psi.transpose()*(m_mKpj*(m_vJointQradDes - m_vJointQrad) + m_mKdj*(m_vJointQdotradDes - m_vJointQdotrad));

		if (m_nEndHandshake == 0) {

			m_dHandshakeStartTime += 0.001;

			if (m_dHandshakeStartTime >= 1.0){
				m_dHandshakeStartTime = 1.0;
			}

			const double gain_start = 1.0;
			const double gain_end   = 0.1;

			double handshake_gain =
				gain_start + (gain_end - gain_start) * m_dHandshakeStartTime;

			m_vTorque.segment(26, 7) =
				(m_mActuation_tau_q2psi.transpose() *
				( handshake_gain *
				( m_mKpj * (m_vJointQradDes     - m_vJointQrad)
				+ m_mKdj * (m_vJointQdotradDes  - m_vJointQdotrad) )
				)).segment(26, 7);
		}
		if (m_nMotionIndex != 6 && m_nEndHandshake == 0 && m_nMotionStep == 3){
			m_dHandshakeEndTime += 0.001;

			if (m_dHandshakeEndTime >= 0.5)
				m_dHandshakeEndTime = 0.5;

			const double gain_start = 0.1;
			const double gain_end   = 1.0;

			double handshake_gain =
				gain_start + (gain_end - gain_start) * m_dHandshakeEndTime;

			m_vTorque.segment(26, 7) =
				(m_mActuation_tau_q2psi.transpose() *
				( handshake_gain *
				( m_mKpj * (m_vJointQradDes     - m_vJointQrad)
				+ m_mKdj * (m_vJointQdotradDes  - m_vJointQdotrad) )
				)).segment(26, 7);
		}
		if(m_dHandshakeEndTime>=0.5){
			m_nEndHandshake = 999;
			m_dHandshakeEndTime = 0.0;
			m_dHandshakeStartTime = 0.0;
		}


		// PACE 용으로 RL 게인 사용 중
		// m_vTorque = m_mActuation_tau_q2psi.transpose()*(m_mKpRLj*(m_vJointQradDes - m_vJointQrad) + m_mKdRLj*(m_vJointQdotradDes - m_vJointQdotrad));
		// m_vTorque = (m_mKpj*(m_vMotorThetaradDes - m_vMotorThetarad) + m_mKdj*(m_vMotorThetadotradDes - m_vMotorThetadotrad));

		// for(int i = 0; i<33; i++){
		// 	cout<<"num "<<i<<endl;
		// 	cout<<m_mKpj(i,i)<<endl;
		// 	cout<<m_mKdj(i,i)<<endl;
		// 	cout<<m_vJointQradDes(i)<<endl;
		// 	cout<<m_vJointQrad(i)<<endl;
		// 	cout<<m_vJointQdotradDes(i)<<endl;
		// 	cout<<m_vJointQdotrad(i)<<endl;
		// }



		m_vRLTorque = m_mKpRLj*(m_vRLJointQradDes - m_vJointQrad) + m_mKdRLj*(m_vRLJointQdotradDes - m_vJointQdotrad);
		// cout<<"m_vRLTorque "<<m_vRLTorque<<endl;
		for(int i = 0; i < m_nJDoF; i++){
			if (abs(m_vRLTorque(i)) > m_vJointTorqueLimitNm(i)){
				m_vRLTorque(i) = std::clamp(m_vRLTorque(i), -m_vJointTorqueLimitNm(i), m_vJointTorqueLimitNm(i));
			}
		}
		m_vRLTorque = m_mActuation_tau_q2psi.transpose()*m_vRLTorque;
		if(!m_bRL){
			m_vRLTorque.setZero();
		}


		if(!m_bINIT){
			// wrist parallel mechanism torque conversion

			//	(원본)
			// left wrist
			m_vTorque.segment(L_wrist_front_D, 2) = -WristParallelMechanism.compute_linear_force_L(
			m_vMotorThetarad(L_wrist_front_D) * 0.001,
			m_vMotorThetarad(L_wrist_back_D) * 0.001, m_vTorque(L_wrist_front_D),
			m_vTorque(L_wrist_back_D));
			// // right wrist
			m_vTorque.segment(R_wrist_front_D, 2) = WristParallelMechanism.compute_linear_force_R(
			m_vMotorThetarad(R_wrist_front_D) * 0.001,
			m_vMotorThetarad(R_wrist_back_D) * 0.001, m_vTorque(R_wrist_front_D),
			m_vTorque(R_wrist_back_D));

			// left wrist
			m_vRLTorque.segment(L_wrist_front_D, 2) = -WristParallelMechanism.compute_linear_force_L(
			m_vMotorThetarad(L_wrist_front_D) * 0.001,
			m_vMotorThetarad(L_wrist_back_D) * 0.001, m_vRLTorque(L_wrist_front_D),
			m_vRLTorque(L_wrist_back_D));
			// right wrist
			m_vRLTorque.segment(R_wrist_front_D, 2) = WristParallelMechanism.compute_linear_force_R(
			m_vMotorThetarad(R_wrist_front_D) * 0.001,
			m_vMotorThetarad(R_wrist_back_D) * 0.001, m_vRLTorque(R_wrist_front_D),
			m_vRLTorque(R_wrist_back_D));

		
		}
			
		// toe tau offset (@@@@@)
		// m_vTorque(6) = 3;
		// m_vTorque(13) = -3;

		if(m_bCONTROL){
			// 재완
			// m_vTorque(0) = 0.0;	//hip yaw
			// m_vTorque(1) = 0.0;	//hip roll
			// m_vTorque(2) = 0.0;	//hip pitch
			// m_vTorque(3) = 0.0;	//knee
			// m_vTorque(4) = 0.0;	//ankle1
			// m_vTorque(5) = 0.0;	//ankle2
			// m_vTorque(6) = 0.0;	//toe

			// m_vTorque(7) = 0.0;	//hip yaw
			// m_vTorque(8) = 0.0;	//hip roll
			// m_vTorque(9) = 0.0;	//hip pitchs
			// m_vTorque(10) = 0.0;	//knees
			// m_vTorque(11) = 0.0;	//ankle1
			// m_vTorque(12) = 0.0;	//ankle2
			// m_vTorque(13) = 0.0;	//toe

			// m_vTorque(14) = 0.0;	//waist roll
			// m_vTorque(15) = 0.0;	//waist 1
			// m_vTorque(16) = 0.0;	//waist 2
			// m_vTorque(17) = 0.0; //head pitch
			// m_vTorque(18) = 0.0; //head yaw

			// m_vTorque(19) = 0.0; //left shoulder pitch
			// m_vTorque(20) = 0.0; //left shoulder roll
			// m_vTorque(21) = 0.0; //left shoulder yaw
			// m_vTorque(22) = 0.0; //left elbow
			// m_vTorque(23) = 0.0; //left wrist 1(roll)
			// m_vTorque(24) = 0.0; //left wrist 2 (pitch)
			// m_vTorque(25) = 0.0; //left wrist yaw

			// m_vTorque(26) = 0.0; //right shoulder pitch
			// m_vTorque(27) = 0.0; //right shoulder roll
			// m_vTorque(28) = 0.0; //right shoulder yaw
			// m_vTorque(29) = 0.0; //right elbow
			// m_vTorque(30) = 0.0; //right wrist 1 (roll)
			// m_vTorque(31) = 0.0; //right wrist 2 (pitch)
			// m_vTorque(32) = 0.0; //right wrist yaw
		}
	}

	// debug
	if(m_dTime == m_dTimePrev){
		m_nCountTimeErr ++;
	}

	m_dTimePrev = m_dTime;
	if(m_nCoutNum == 100){
		if(!m_bTorqueOff){
			cout<<"current time : "<<m_dTime<<endl;
			if(m_nCountTimeErr != 0){
				cout<<"tiem err count : "<<m_nCountTimeErr<<endl;	
			}
			if(m_dControlStartTime != 0){
				cout<<"control start time : "<<m_dControlStartTime<<endl;
			}
			cout<<"control time : "<<m_dControlTime<<endl;

			cout<<"control mode : "<<m_iControlMode<<endl;
			// cout<<"joint "<<m_vJointQrad.size()<<endl;
			// cout<<m_vJointQrad<<endl;
			// cout<<m_vJointQrad.segment(7,7)<<endl;
			// cout<<m_vJointQrad.segment(14,5)<<endl;
			// cout<<m_vJointQrad.segment(19,7).transpose()<<endl;
			// cout<<m_vJointQrad.segment(26,7).transpose()<<endl;

			// cout<<"joint des "<<m_vJointQradDes.size()<<endl;
			// cout<<m_vJointQradDes.segment(0,7).transpose()<<endl;
			// cout<<m_vJointQradDes.segment(7,7).transpose()<<endl;
			// cout<<m_vJointQradDes.segment(14,5).transpose()<<endl;
			// cout<<m_vJointQradDes.segment(19,7).transpose()<<endl;
			// cout<<m_vJointQradDes.segment(26,7).transpose()<<endl;

			// cout<<"jointdot "<<m_vJointQdotrad.size()<<endl;
			// cout<<m_vJointQdotrad.segment(0,7).transpose()<<endl;
			// cout<<m_vJointQdotrad.segment(7,7).transpose()<<endl;
			// cout<<m_vJointQdotrad.segment(14,5).transpose()<<endl;
			// cout<<m_vJointQdotrad.segment(19,7).transpose()<<endl;
			// cout<<m_vJointQdotrad.segment(26,7).transpose()<<endl<<endl;

			// cout<<"motor "<<m_vMotorThetarad.size()<<endl;
			// cout<<m_vMotorThetarad.segment(0,7).transpose()<<endl;
			// cout<<m_vMotorThetarad.segment(7,7).transpose()<<endl;
			// cout<<m_vMotorThetarad.segment(14,5).transpose()<<endl;
			// cout<<m_vMotorThetarad.segment(19,7).transpose()<<endl;
			// cout<<m_vMotorThetarad.segment(26,7).transpose()<<endl;
			// cout<<"motordot "<<m_vMotorThetadotrad.size()<<endl;
			// cout<<m_vMotorThetadotrad.segment(0,7).transpose()<<endl;
			// cout<<m_vMotorThetadotrad.segment(7,7).transpose()<<endl;
			// cout<<m_vMotorThetadotrad.segment(14,5).transpose()<<endl;
			// cout<<m_vMotorThetadotrad.segment(19,7).transpose()<<endl;
			// cout<<m_vMotorThetadotrad.segment(26,7).transpose()<<endl<<endl;

			// cout<<"motor abs "<<m_vMotorThetaAbsrad.size()<<endl;
			// cout<<m_vMotorThetaAbsrad.segment(0,7).transpose()<<endl;
			// cout<<m_vMotorThetaAbsrad.segment(7,7).transpose()<<endl;
			// cout<<m_vMotorThetaAbsrad.segment(14,5).transpose()<<endl;
			// cout<<m_vMotorThetaAbsrad.segment(19,7).transpose()<<endl;
			// cout<<m_vMotorThetaAbsrad.segment(26,7).transpose()<<endl;

			// cout<<"m_vTorque "<<m_vTorque.size()<<endl;
			// cout<<m_vTorque.segment(0,7).transpose()<<endl;
			// cout<<m_vTorque.segment(7,7).transpose()<<endl;
			// cout<<m_vTorque.segment(14,5).transpose()<<endl;
			// cout<<m_vTorque.segment(19,7).transpose()<<endl;
			// cout<<m_vTorque.segment(26,7).transpose()<<endl<<endl;

			// cout<<"RL joint des "<<m_vRLJointQradDes.size()<<endl;
			// cout<<m_vRLJointQradDes.segment(0,7).transpose()<<endl;
			// cout<<m_vRLJointQradDes.segment(7,7).transpose()<<endl;
			// cout<<m_vRLJointQradDes.segment(14,5).transpose()<<endl;

			// cout<<"m_vRLTorque "<<m_vRLTorque.size()<<endl;
			// cout<<m_vRLTorque.segment(0,7).transpose()<<endl;
			// cout<<m_vRLTorque.segment(7,7).transpose()<<endl;
			// cout<<m_vRLTorque.segment(14,5).transpose()<<endl;

			// cout<<"m_dNeckPitchDes"<<endl<<m_dNeckPitchDes<<endl;
			// cout<<"m_dNeckYawDes"<<endl<<m_dNeckYawDes<<endl;

			if(m_bCONTROL && m_bMotion){
				cout<<"Motion Index : "<<m_nMotionIndex<<" ( "<<m_nMotionIndexNum<<", "<<m_pCurrentMotionData->size()<<" )"<<endl;
			}

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
	// PACE
// m_vTorque.setZero();
	// m_vTorque(0) = 0.0;	//hip yaw
	// m_vTorque(1) = 0.0;	//hip roll
	// m_vTorque(2) = 0.0;	//hip pitch
	// m_vTorque(3) = 0.0;	//knee
	m_vTorque(4) = 0.0;	//ankle1
	m_vTorque(5) = 0.0;	//ankle2
	m_vTorque(6) = 0.0;	//toe

	// m_vTorque(7) = 0.0;	//hip yaw
	// m_vTorque(8) = 0.0;	//hip roll
	// m_vTorque(9) = 0.0;	//hip pitchs
	// m_vTorque(10) = 0.0;	//knees
	m_vTorque(11) = 0.0;	//ankle1
	m_vTorque(12) = 0.0;	//ankle2
	m_vTorque(13) = 0.0;	//toe

	// m_vTorque(14) = 0.0;	//waist roll
	m_vTorque(15) = 0.0;	//waist 1
	m_vTorque(16) = 0.0;	//waist 2
	m_vTorque(17) = 0.0; //head pitch
	m_vTorque(18) = 0.0; //head yaw

	// m_vTorque(19) = 0.0; //left shoulder pitch
	// m_vTorque(20) = 0.0; //left shoulder roll
	// m_vTorque(21) = 0.0; //left shoulder yaw
	// m_vTorque(22) = 0.0; //left elbow
	// m_vTorque(23) = 0.0; //left wrist 1
	// m_vTorque(24) = 0.0; //left wrist 2
	// m_vTorque(25) = 0.0; //left wrist yaw

	// m_vTorque(26) = 0.0; //right shoulder pitch
	// m_vTorque(27) = 0.0; //right shoulder roll
	// m_vTorque(28) = 0.0; //right shoulder yaw
	// m_vTorque(29) = 0.0; //right elbow
	// m_vTorque(30) = 0.0; //right wrist 1
	// m_vTorque(31) = 0.0; //right wrist 2
	// m_vTorque(32) = 0.0; //right wrist yaw
}

Vector<double,NUM_OF_JOINTS> CJointControl::psi2q(Vector<double, NUM_OF_JOINTS> vPsi)
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

	//head
	m_vQ_psi2q(17) = vPsi(17);	//pitch
	m_vQ_psi2q(18) = vPsi(18);	//yaw


	//	(원본)
	//left arm
	m_vQ_psi2q(19) = vPsi(19);	//shoulder pitch
	m_vQ_psi2q(20) = vPsi(20);	//shoulder roll
	m_vQ_psi2q(21) = vPsi(21);	//shoulder yaw

	m_vQ_psi2q(22) = vPsi(22);	//elbow

	m_vQ_psi2q(23) = vPsi(25);	//wrist yaw
	WristParallelMechanism.ForwardKinematics_L(
		m_vMotorThetarad(L_wrist_front_D) * 0.001,
		m_vMotorThetarad(L_wrist_back_D) * 0.001);

	m_vQ_psi2q(L_wrist_roll) =
		WristParallelMechanism.getCurrentRoll_L(); //wrist roll
	m_vQ_psi2q(L_wrist_pitch) =
		WristParallelMechanism.getCurrentPitch_L(); //wrist pitch
	
	//right arm
	m_vQ_psi2q(26) = vPsi(26);	//shoulder pitch
	m_vQ_psi2q(27) = vPsi(27);	//shoulder roll
	m_vQ_psi2q(28) = vPsi(28);	//shoulder yaw

	m_vQ_psi2q(29) = vPsi(29);	//elbow

	m_vQ_psi2q(30) = vPsi(32);
	WristParallelMechanism.ForwardKinematics_R(
		m_vMotorThetarad(R_wrist_front_D) * 0.001,
		m_vMotorThetarad(R_wrist_back_D) * 0.001);

	m_vQ_psi2q(R_wrist_roll) = 
		WristParallelMechanism.getCurrentRoll_R();
	m_vQ_psi2q(R_wrist_pitch) =
		WristParallelMechanism.getCurrentPitch_R();

	//	(임시)
	//right arm
	// m_vQ_psi2q(19) = vPsi(19);	//shoulder pitch
	// m_vQ_psi2q(20) = vPsi(20);	//shoulder roll
	// m_vQ_psi2q(21) = vPsi(21);	//shoulder yaw

	// m_vQ_psi2q(22) = vPsi(22);	//elbow

	// m_vQ_psi2q(23) = vPsi(25);	//wrist yaw
	// WristParallelMechanism.ForwardKinematics_R(
	// 	m_vMotorThetarad(R_wrist_front_D) * 0.001,
	// 	m_vMotorThetarad(R_wrist_back_D) * 0.001);

	// m_vQ_psi2q(R_wrist_roll) = WristParallelMechanism.getCurrentRoll_R();
	// m_vQ_psi2q(R_wrist_pitch) = WristParallelMechanism.getCurrentPitch_R();

	// cout << "m_vQ_psi2q : ";

	// for(int i = 14; i < NUM_OF_JOINTS; i++){
	// 	cout << m_vQ_psi2q(i) << " ";
	// }

	// cout << endl;

	return m_vQ_psi2q; 
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


Eigen::Matrix<double, NUM_OF_JOINTS, NUM_OF_JOINTS> CJointControl::compute_tau_q2psi(Vector<double, NUM_OF_JOINTS> vX){
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
	m_mJ_jacobian(17, 17) = 1.0;
	m_mJ_jacobian(18, 18) = 1.0;

	// 일단 모터 기준임 포워드 풀어서 반영해야함!!!
	// left arm
	m_mJ_jacobian(23, 23) = 0.0;
	m_mJ_jacobian(24, 24) = 0.0;
	m_mJ_jacobian(25, 25) = 0.0;

	m_mJ_jacobian(23, 25) = 1.0;
	m_mJ_jacobian(24, 23) = 1.0;
	m_mJ_jacobian(25, 24) = 1.0;
	// right arm
	m_mJ_jacobian(30, 30) = 0.0;
	m_mJ_jacobian(31, 31) = 0.0;
	m_mJ_jacobian(32, 32) = 0.0;

	m_mJ_jacobian(30, 32) = 1.0;
	m_mJ_jacobian(31, 30) = 1.0;
	m_mJ_jacobian(32, 31) = 1.0;

	return m_mJ_jacobian;
}

void CJointControl::Initialize()
{	
	cout<<"Joint Control Initialize!"<<endl;
    /* public Robot State */
	m_vTorque.setZero();
	m_vMotorThetarad.setZero();
	m_vMotorThetaradDes.setZero();
	m_vPreMotorThetaradDes.setZero();
	m_vMotorThetadotrad.setZero();
	m_vMotorThetadotradDes.setZero();
	m_vPrMotorThetadotradDes.setZero();
	m_vMotorThetaAbsrad.setZero();
	m_vJointQrad.setZero();
	m_vJointQradDes.setZero();
	m_vPreJointQrad.setZero();
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
	m_dTimePrev = 0.0;
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
	m_bInitWrist = false;

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
	JointTrajectory.set_size(NUM_OF_JOINTS);
	MotorTrajectory.set_size(NUM_OF_JOINTS);
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
	m_vhomeposition.segment(0,m_nJDoF) <<	0.03, 0.03, -0.09, +0.38, -0.33, 0.0, 0.0,
											-0.03, -0.03, +0.09, -0.38, +0.33, 0.0, 0.0,
											0.0, 0.0, 0.0, 0.09, 0.0,
											0.05, 0.2, 0.18, -0.55, 0.0, 0.0, 0.0,
											-0.05, -0.2, -0.18, 0.55, 0.0, 0.0, 0.0;	
	m_nCoutNum = 0;
	m_nCoutVelLimitNum = 0;
	m_nCountTimeErr = 0;
	m_vTorqueLimitNm.setZero();

	double torquelimitgain = 0.4;
	m_vTorqueLimitNm.segment(0, m_nJDoF) << 70.0, 70.0, 150.0, 150.0, 70.0, 70.0, 20.0,
											70.0, 70.0, 150.0, 150.0, 70.0, 70.0, 20.0,
											100.0, 100.0, 100.0, 13.0, 6.0, 
											70.0, 70.0, 30.0, 30.0, 0.5, 0.5, 20.0,
											70.0, 70.0, 30.0, 30.0, 0.5, 0.5, 20.0;
	m_vTorqueLimitNm = torquelimitgain * m_vTorqueLimitNm;

	m_vTorqueLimitNm(23) = 0.5;
	m_vTorqueLimitNm(24) = 0.5;
	m_vTorqueLimitNm(30) = 0.5;
	m_vTorqueLimitNm(31) = 0.5;

	m_vTorqueLimitNm.tail(40).setConstant(10000.0); //for hand motors

	m_vJointTorqueLimitNm.segment(0, m_nJDoF) << 70, 70, 180, 180, 60, 60, 20,
												 70, 70, 180, 180, 60, 60, 20,
												 100.0, 100.0, 100.0, 13.0, 6.0, 
												 70.0, 70.0, 30.0, 30.0, 0.5, 0.5, 20.0,
												 70.0, 70.0, 30.0, 30.0, 0.5, 0.5, 20.0;
											
	m_vJointQradUpLimit.setZero();
	m_vJointQradUpLimit.segment(0, m_nJDoF) << 0.3491, 0.3491, 0.5236, 1.91986, 0.87266, 0.3491, 0.3491,
												0.7854, 0.3491, 1.7453, 0.0, 0.5236, 0.3491, 1.0472,
												0.523599, 0.523599, 0.7854, 0.2967, 1.2217,
												0.8727, 3.0543, 1.6581, 0.0175, 1.1345, 0.5236*0.95, 0.7854*0.95,
												3.1416, -0.0873, 1.6581, 1.6581, 1.1345, 0.6632*0.95, 0.7854*0.95;
	m_vJointQradLowLimit.setZero();
	m_vJointQradLowLimit.segment(0, m_nJDoF) << -0.7854, -0.3491, -1.7453, 0.0, -0.5236, -0.3491, -1.0472,
												-0.3491, -0.3491, -0.5236, -1.91986, -0.87266, -0.3491, -0.3491,
												-0.523599, -0.523599, -0.174533, -0.7854, -1.2217,
												-3.1416, 0.0873, -1.6581, -1.6581, -1.1345, -0.6632*0.95, -0.7854*0.95,
												-0.8727, -3.0543, -1.6581, -0.0175, -1.1345, -0.5236*0.95, -0.7854*0.95;

	m_vMotorThetaradUpLimit.setZero();
	m_vMotorThetaradUpLimit.segment(0, m_nJDoF) << 0.3491, 0.3491, 0.5236, 1.9111, 1.2741, 0.5, 0.7854,
													0.7854, 0.3491, 1.7453, 0.0175, 0.6109, 1.513, 1.1868,
													0.3491, 2.0944, 0.6981, 0.2967, 1.2217,
													3.141, 0.261, 1.658, 1.658, 28, 28, 1.745,
													1.5, 1.5, 1.5, 2.0, 1.5, 1.5, 1.5;

	m_vMotorThetaradLowLimit.setZero();
	m_vMotorThetaradLowLimit.segment(0, m_nJDoF) << -0.7854, -0.3491, -1.7453, -0.0175, -0.6109, -1.513,-1.1868,
													-0.3491, -0.3491, -0.5236, -1.9111, -1.2741, -0.5, -0.7854,
													-0.3491, -0.6981, -2.0944, -0.7854, -1.2217,
													-0.872, -3.054, -1.658, -0.349, -13, -13, -1.745,
													-1.5, -1.5, -1.5, -2.0, -1.5, -1.5, -1.5;

	// cout<<"m_vMotorThetaradLowLimit"<<m_vMotorThetaradLowLimit.transpose()<<endl;

	m_vJointQdotradLimit.setZero();
	m_vJointQdotradLimit.segment(0, m_nJDoF) << 10, 10, 10, 10, 30, 30, 50.0,
											 	  10, 10, 10, 10, 30, 30, 50.0,
											 	  5.0, 5.0, 5.0, 5.0, 5.0,
											 	  0.8, 0.8, 8.99, 5.0, 5.0, 30, 30,
											 	  0.8, 0.8, 8.99, 5.0, 5.0, 30, 30;
	m_vMotorThetadotradLimit.setZero();
	m_vMotorThetadotradLimit.segment(0, m_nJDoF) << 10.99, 10.99, 10.99, 10.99, 10.99, 10.99, 50.0,
											 	  	  10.99, 10.99, 10.99, 10.99, 10.99, 10.99, 50.0,
											 	  	  5.0, 5.0, 5.0, 5.0, 5.0,
											 	  	  0.8, 0.8, 1.5, 5.0, 30, 30, 5.0,
											 	  	  0.8, 0.8, 1.5, 5.0, 30, 30, 5.0;
													  
	m_bTorqueOff = false;
	m_bStandStill = false;
	m_dPreviousComputationTime = 0.0;
	m_bMPPItimeout = true;
	m_dMPPItimeoutstep = 0;
	m_ldMPPItimes = 0.0;

    /* Pre-allocated Variables for Functions */
    m_vQ_psi2q.setZero();
    m_dTheta2_psi2q = 0.0; m_dBeta_psi2q = 0.0; m_dLambda_psi2q = 0.0; m_dL_squared_psi2q = 0.0;
    m_dTheta1_psi2q = 0.0; m_dTheta41_psi2q = 0.0; m_dBeta1_psi2q = 0.0; m_dLambda1_psi2q = 0.0;
    m_dTheta42_psi2q = 0.0; m_dBeta2_psi2q = 0.0; m_dLambda2_psi2q = 0.0; m_dL__squared_psi2q = 0.0;

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

	// 유선 데이터 로드 실행
    // load_segmented_data(PATH_NAMES, PATH_THETA, PATH_VEL, ACTIVE_JOINT_INDEX);

    // precompute_trajectory();

	// 모션 데이터 로드
	ReadMotionCsv(m_sMotionPath0, m_MotionData0);
	ReadMotionCsv(m_sMotionPath1, m_MotionData1);
	ReadMotionCsv(m_sMotionPath2, m_MotionData2);
	ReadMotionCsv(m_sMotionPath3, m_MotionData3);
	ReadMotionCsv(m_sMotionPath4, m_MotionData4);
	ReadMotionCsv(m_sMotionPath5, m_MotionData5);
	ReadMotionCsv(m_sMotionPath6, m_MotionData6);
	ReadMotionCsv(m_sMotionPath7, m_MotionData7);
	ReadMotionCsv(m_sMotionPath8, m_MotionData8);

	NeckTrajectory.set_size(2);
	NeckTrackingTrajectory.set_size(2);

	m_vNeckgoal.setZero();
	m_vNeckdotgoal.setZero();
	m_vPreNeckgoal.setZero();
	m_vPreNeckdotgoal.setZero();

	WaistTrajectory.set_size(3);
	m_vPreWaistgoal.setZero();
	m_vPreWaistdotgoal.setZero();

	m_vWaistgoal.setZero();
	m_vWaistdotgoal.setZero();
	m_dneckmotiontime, m_dneckmotionendtime = 0.0;

	cout<<"Joint Control Initialized"<<endl;
}

void CJointControl::ReadMotionCsv(string path, std::vector<Eigen::VectorXd>& out_data) {
    out_data.clear();
    std::ifstream file(path);
    std::string line;

    if (!file.is_open()) {
        std::cerr << "\033[31m" << "[Error] Cannot open motion file: " << path << "\033[0m" << std::endl;
        return;
    }

    // 1. 헤더 건너뛰기 (0, 1, 2... 인덱스 행)
    std::getline(file, line);

    // 2. 데이터 로드
    // CSV 구조: 
    // 0~6: Left Leg (7)
    // 7~13: Right Leg (7)
    // 14~16: Waist (3)
    // 17~18: Head (2)
    // 19~25: Left Arm (7)
    // 26~45: Left Hand (20) -> IGNORE
    // 46~52: Right Arm (7)
    // 53~72: Right Hand (20) -> IGNORE
	// CSV Right Hand Data Start Index

	const int reorderMap[20] = {
        4, 3, 2, 1, 0,      // Thumb (5)
        8, 7, 6, 5,         // Index (4) -> 5+3 ~ 5+0
        12, 11, 10, 9,      // Middle (4) -> 9+3 ~ 9+0
        16, 15, 14, 13,     // Ring (4) -> 13+3 ~ 13+0
        19, 18, 17          // Little (3) -> 17+2 ~ 17+0
    };

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        if (line.back() == '\r') line.pop_back();

        std::stringstream ss(line);
        std::string cell;
        std::vector<double> raw_values;
        
        // 전체 컬럼 읽기
        while (std::getline(ss, cell, ',')) {
            try {
                raw_values.push_back(std::stod(cell));
            } catch (...) {
                raw_values.push_back(0.0);
            }
        }

        if (raw_values.size() < 53) continue; 

        // 필요한 관절만 추출하여 로봇 state 벡터(33개) 구성
        // 순서: LL(7), RL(7), Waist(3), Head(2), LA(7), RA(7)
        Eigen::VectorXd vec_frame = Eigen::VectorXd::Zero(NUM_OF_JOINTS);
        Eigen::VectorXd vec_frame_temp = Eigen::VectorXd::Zero(NUM_OF_JOINTS);
        
        int target_idx = 0;

        // Helper Lambda for clamping
        auto clamp_val = [&](double val, int idx) {
            if (idx < m_vJointQradLowLimit.size() && idx < m_vJointQradUpLimit.size()) {
                return std::clamp(val, m_vJointQradLowLimit(idx)+0.05, m_vJointQradUpLimit(idx)-0.05);
            }
            return val;
        };

        // Left Leg (0-6)
        for(int i=0; i<7; i++) {
            vec_frame(target_idx) = clamp_val(raw_values[0 + i], target_idx);
            target_idx++;
        }
        
        // Right Leg (7-13)
        for(int i=0; i<7; i++) {
            vec_frame(target_idx) = clamp_val(raw_values[7 + i], target_idx);
            target_idx++;
        }

        // Waist (14-16)
        for(int i=0; i<3; i++) {
            vec_frame(target_idx) = clamp_val(raw_values[14 + i], target_idx);
            target_idx++;
        }

        // Head (17-18)
        for(int i=0; i<2; i++) {
            vec_frame(target_idx) = clamp_val(raw_values[17 + i], target_idx);
            target_idx++;
        }

        // Left Arm (19-25)
        for(int i=0; i<7; i++) {
            vec_frame(target_idx) = clamp_val(raw_values[19 + i], target_idx);
            target_idx++;
        }

        // Right Arm (46-52)
        for(int i=0; i<7; i++) {
            vec_frame(target_idx) = clamp_val(raw_values[46 + i], target_idx);
            target_idx++;
        }

        // Left Hand (26-45)
        for(int i=0; i<20; i++) {
            vec_frame(target_idx) = raw_values[26 + i];
            target_idx++;
        }

		// Right Hand (53-72)
        for(int i=0; i<20; i++) {
            vec_frame(target_idx) = raw_values[53 + i];
            target_idx++;
        }
		
		vec_frame_temp=vec_frame;
		// Left Hand Reordering
		for (int i = 33; i < 53; i++) {
			vec_frame_temp(i) = vec_frame[reorderMap[i-33]+33];
		}

		// Right Hand Reordering
		for (int i = 53; i < 73; i++) {
			vec_frame_temp(i) = vec_frame[reorderMap[i-53]+53];
		}		
		vec_frame.tail(40)=vec_frame_temp.tail(40);

        out_data.push_back(vec_frame);
    }

    file.close();
    std::cout << "[MotionLoader] Loaded " << out_data.size() << " frames from " << path << std::endl;

	if (!out_data.empty()) {
        std::cout << ">>> First Frame Data (" << path << "):" << std::endl;
        std::cout << out_data[0].transpose() << std::endl;
    }
}

void CJointControl::SetDesiredPosition(const std::array<float, 17>& qDes)
{
	for (size_t i = 0; i < qDes.size(); ++i) {
		m_vRLJointQradDes(i) = static_cast<double>(qDes[i]);
	}
}