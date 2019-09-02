#!/bin/bash

#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=120:00:00
#SBATCH --array=0-52

GAMES=(AirRaidDeterministic-v4 AlienDeterministic-v4 AmidarDeterministic-v4 AssaultDeterministic-v4
AsterixDeterministic-v4 AsteroidsDeterministic-v4 AtlantisDeterministic-v4 BankHeistDeterministic-v4
BattleZoneDeterministic-v4 BeamRiderDeterministic-v4 BerzerkDeterministic-v4 BowlingDeterministic-v4
BreakoutDeterministic-v4 CarnivalDeterministic-v4 CentipedeDeterministic-v4 ChopperCommandDeterministic-v4
CrazyClimberDeterministic-v4 DemonAttackDeterministic-v4 DoubleDunkDeterministic-v4 ElevatorActionDeterministic-v4
FishingDerbyDeterministic-v4 FreewayDeterministic-v4 FrostbiteDeterministic-v4 GopherDeterministic-v4 GravitarDeterministic-v4
JamesbondDeterministic-v4 JourneyEscapeDeterministic-v4 KangarooDeterministic-v4 KrullDeterministic-v4 KungFuMasterDeterministic-v4
MontezumaRevengeDeterministic-v4 MsPacmanDeterministic-v4 NameThisGameDeterministic-v4 PhoenixDeterministic-v4 PitfallDeterministic-v4
PooyanDeterministic-v4 PrivateEyeDeterministic-v4 RiverraidDeterministic-v4 RoadRunnerDeterministic-v4
RobotankDeterministic-v4 SeaquestDeterministic-v4 SkiingDeterministic-v4 SolarisDeterministic-v4
SpaceInvadersDeterministic-v4 StarGunnerDeterministic-v4 TennisDeterministic-v4
TimePilotDeterministic-v4 TutankhamDeterministic-v4 UpNDownDeterministic-v4 VentureDeterministic-v4
VideoPinballDeterministic-v4 WizardOfWorDeterministic-v4 ZaxxonDeterministic-v4)


source activate dl
KERAS_BACKEND=tensorflow

policy_mode="online"
agent="dqv"
episodes=100000

nohup python choose_rl_ensemble.py --policy_mode $policy_mode --agent $agent --source_game ${GAMES[$SLURM_ARRAY_TASK_ID]} --episodes $episodes
