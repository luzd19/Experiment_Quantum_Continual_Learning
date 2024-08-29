OPENQASM 2.0;
include "qelib1.inc";
qreg q[18];
creg c[18];
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
rx(11.0) q[10];
rx(12.0) q[11];
rx(13.0) q[12];
rx(14.0) q[13];
rx(15.0) q[14];
rx(16.0) q[15];
rx(17.0) q[16];
rx(18.0) q[17];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17];
rz(19.0) q[0];
rz(20.0) q[1];
rz(21.0) q[2];
rz(22.0) q[3];
rz(23.0) q[4];
rz(24.0) q[5];
rz(25.0) q[6];
rz(26.0) q[7];
rz(27.0) q[8];
rz(28.0) q[9];
rz(29.0) q[10];
rz(30.0) q[11];
rz(31.0) q[12];
rz(32.0) q[13];
rz(33.0) q[14];
rz(34.0) q[15];
rz(35.0) q[16];
rz(36.0) q[17];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17];
rx(37.0) q[0];
rx(38.0) q[1];
rx(39.0) q[2];
rx(40.0) q[3];
rx(41.0) q[4];
rx(42.0) q[5];
rx(43.0) q[6];
rx(44.0) q[7];
rx(45.0) q[8];
rx(46.0) q[9];
rx(47.0) q[10];
rx(48.0) q[11];
rx(49.0) q[12];
rx(50.0) q[13];
rx(51.0) q[14];
rx(52.0) q[15];
rx(53.0) q[16];
rx(54.0) q[17];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17];
cx q[0],q[1];
cx q[2],q[3];
cx q[4],q[5];
cx q[6],q[7];
cx q[8],q[9];
cx q[10],q[11];
cx q[12],q[13];
cx q[14],q[15];
cx q[16],q[17];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17];
cx q[1],q[2];
cx q[3],q[4];
cx q[5],q[6];
cx q[7],q[8];
cx q[9],q[10];
cx q[11],q[12];
cx q[13],q[14];
cx q[15],q[16];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17];
rx(55.0) q[0];
rx(56.0) q[1];
rx(57.0) q[2];
rx(58.0) q[3];
rx(59.0) q[4];
rx(60.0) q[5];
rx(61.0) q[6];
rx(62.0) q[7];
rx(63.0) q[8];
rx(64.0) q[9];
rx(65.0) q[10];
rx(66.0) q[11];
rx(67.0) q[12];
rx(68.0) q[13];
rx(69.0) q[14];
rx(70.0) q[15];
rx(71.0) q[16];
rx(72.0) q[17];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17];
rz(73.0) q[0];
rz(74.0) q[1];
rz(75.0) q[2];
rz(76.0) q[3];
rz(77.0) q[4];
rz(78.0) q[5];
rz(79.0) q[6];
rz(80.0) q[7];
rz(81.0) q[8];
rz(82.0) q[9];
rz(83.0) q[10];
rz(84.0) q[11];
rz(85.0) q[12];
rz(86.0) q[13];
rz(87.0) q[14];
rz(88.0) q[15];
rz(89.0) q[16];
rz(90.0) q[17];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17];
rx(91.0) q[0];
rx(92.0) q[1];
rx(93.0) q[2];
rx(94.0) q[3];
rx(95.0) q[4];
rx(96.0) q[5];
rx(97.0) q[6];
rx(98.0) q[7];
rx(99.0) q[8];
rx(100.0) q[9];
rx(101.0) q[10];
rx(102.0) q[11];
rx(103.0) q[12];
rx(104.0) q[13];
rx(105.0) q[14];
rx(106.0) q[15];
rx(107.0) q[16];
rx(108.0) q[17];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17];
cx q[0],q[1];
cx q[2],q[3];
cx q[4],q[5];
cx q[6],q[7];
cx q[8],q[9];
cx q[10],q[11];
cx q[12],q[13];
cx q[14],q[15];
cx q[16],q[17];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17];
cx q[1],q[2];
cx q[3],q[4];
cx q[5],q[6];
cx q[7],q[8];
cx q[9],q[10];
cx q[11],q[12];
cx q[13],q[14];
cx q[15],q[16];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17];
rx(109.0) q[0];
rx(110.0) q[1];
rx(111.0) q[2];
rx(112.0) q[3];
rx(113.0) q[4];
rx(114.0) q[5];
rx(115.0) q[6];
rx(116.0) q[7];
rx(117.0) q[8];
rx(118.0) q[9];
rx(119.0) q[10];
rx(120.0) q[11];
rx(121.0) q[12];
rx(122.0) q[13];
rx(123.0) q[14];
rx(124.0) q[15];
rx(125.0) q[16];
rx(126.0) q[17];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17];
rz(127.0) q[0];
rz(128.0) q[1];
rz(129.0) q[2];
rz(130.0) q[3];
rz(131.0) q[4];
rz(132.0) q[5];
rz(133.0) q[6];
rz(134.0) q[7];
rz(135.0) q[8];
rz(136.0) q[9];
rz(137.0) q[10];
rz(138.0) q[11];
rz(139.0) q[12];
rz(140.0) q[13];
rz(141.0) q[14];
rz(142.0) q[15];
rz(143.0) q[16];
rz(144.0) q[17];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17];
rx(145.0) q[0];
rx(146.0) q[1];
rx(147.0) q[2];
rx(148.0) q[3];
rx(149.0) q[4];
rx(150.0) q[5];
rx(151.0) q[6];
rx(152.0) q[7];
rx(153.0) q[8];
rx(154.0) q[9];
rx(155.0) q[10];
rx(156.0) q[11];
rx(157.0) q[12];
rx(158.0) q[13];
rx(159.0) q[14];
rx(160.0) q[15];
rx(161.0) q[16];
rx(162.0) q[17];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17];
cx q[0],q[1];
cx q[2],q[3];
cx q[4],q[5];
cx q[6],q[7];
cx q[8],q[9];
cx q[10],q[11];
cx q[12],q[13];
cx q[14],q[15];
cx q[16],q[17];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17];
cx q[1],q[2];
cx q[3],q[4];
cx q[5],q[6];
cx q[7],q[8];
cx q[9],q[10];
cx q[11],q[12];
cx q[13],q[14];
cx q[15],q[16];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17];
rx(163.0) q[0];
rx(164.0) q[1];
rx(165.0) q[2];
rx(166.0) q[3];
rx(167.0) q[4];
rx(168.0) q[5];
rx(169.0) q[6];
rx(170.0) q[7];
rx(171.0) q[8];
rx(172.0) q[9];
rx(173.0) q[10];
rx(174.0) q[11];
rx(175.0) q[12];
rx(176.0) q[13];
rx(177.0) q[14];
rx(178.0) q[15];
rx(179.0) q[16];
rx(180.0) q[17];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17];
rz(181.0) q[0];
rz(182.0) q[1];
rz(183.0) q[2];
rz(184.0) q[3];
rz(185.0) q[4];
rz(186.0) q[5];
rz(187.0) q[6];
rz(188.0) q[7];
rz(189.0) q[8];
rz(190.0) q[9];
rz(191.0) q[10];
rz(192.0) q[11];
rz(193.0) q[12];
rz(194.0) q[13];
rz(195.0) q[14];
rz(196.0) q[15];
rz(197.0) q[16];
rz(198.0) q[17];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17];
rx(199.0) q[0];
rx(200.0) q[1];
rx(201.0) q[2];
rx(202.0) q[3];
rx(203.0) q[4];
rx(204.0) q[5];
rx(205.0) q[6];
rx(206.0) q[7];
rx(207.0) q[8];
rx(208.0) q[9];
rx(209.0) q[10];
rx(210.0) q[11];
rx(211.0) q[12];
rx(212.0) q[13];
rx(213.0) q[14];
rx(214.0) q[15];
rx(215.0) q[16];
rx(216.0) q[17];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17];
cx q[0],q[1];
cx q[2],q[3];
cx q[4],q[5];
cx q[6],q[7];
cx q[8],q[9];
cx q[10],q[11];
cx q[12],q[13];
cx q[14],q[15];
cx q[16],q[17];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17];
cx q[1],q[2];
cx q[3],q[4];
cx q[5],q[6];
cx q[7],q[8];
cx q[9],q[10];
cx q[11],q[12];
cx q[13],q[14];
cx q[15],q[16];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17];
measure q[8] -> c[8];