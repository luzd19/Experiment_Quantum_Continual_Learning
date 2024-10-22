OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
creg c[10];
rx(1.0) q[0];
rx(2.0) q[1];
rx(3.0) q[2];
rx(4.0) q[3];
rx(5.0) q[4];
rx(6.0) q[5];
rx(7.0) q[6];
rx(8.0) q[7];
rx(9.0) q[8];
rx(10.0) q[9];
rz(11.0) q[0];
rz(12.0) q[1];
rz(13.0) q[2];
rz(14.0) q[3];
rz(15.0) q[4];
rz(16.0) q[5];
rz(17.0) q[6];
rz(18.0) q[7];
rz(19.0) q[8];
rz(20.0) q[9];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9];
rx(21.0) q[0];
rx(22.0) q[1];
rx(23.0) q[2];
rx(24.0) q[3];
rx(25.0) q[4];
rx(26.0) q[5];
rx(27.0) q[6];
rx(28.0) q[7];
rx(29.0) q[8];
rx(30.0) q[9];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9];
rx(31.0) q[0];
rx(32.0) q[1];
rx(33.0) q[2];
rx(34.0) q[3];
rx(35.0) q[4];
rx(36.0) q[5];
rx(37.0) q[6];
rx(38.0) q[7];
rx(39.0) q[8];
rx(40.0) q[9];
rz(41.0) q[0];
rz(42.0) q[1];
rz(43.0) q[2];
rz(44.0) q[3];
rz(45.0) q[4];
rz(46.0) q[5];
rz(47.0) q[6];
rz(48.0) q[7];
rz(49.0) q[8];
rz(50.0) q[9];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9];
rx(51.0) q[0];
rx(52.0) q[1];
rx(53.0) q[2];
rx(54.0) q[3];
rx(55.0) q[4];
rx(56.0) q[5];
rx(57.0) q[6];
rx(58.0) q[7];
rx(59.0) q[8];
rx(60.0) q[9];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9];
rx(61.0) q[0];
rx(62.0) q[1];
rx(63.0) q[2];
rx(64.0) q[3];
rx(65.0) q[4];
rx(66.0) q[5];
rx(67.0) q[6];
rx(68.0) q[7];
rx(69.0) q[8];
rx(70.0) q[9];
rz(71.0) q[0];
rz(72.0) q[1];
rz(73.0) q[2];
rz(74.0) q[3];
rz(75.0) q[4];
rz(76.0) q[5];
rz(77.0) q[6];
rz(78.0) q[7];
rz(79.0) q[8];
rz(80.0) q[9];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9];
rx(81.0) q[0];
rx(82.0) q[1];
rx(83.0) q[2];
rx(84.0) q[3];
rx(85.0) q[4];
rx(86.0) q[5];
rx(87.0) q[6];
rx(88.0) q[7];
rx(89.0) q[8];
rx(90.0) q[9];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9];
measure q[0] -> c[0];
