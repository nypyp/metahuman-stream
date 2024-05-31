var CurrentArea = "All";
var EtsChart;
var TcoptChart;
var PcpChart;
var QcstChart;
var CcChart;
var GHmapChart;

var sdsdata = {
	spaceTime: 1, //数据请求间隔时间单位-分钟
	init: function init() {
		var wh = $(window).height();
		var th = $("header").height();
		$('#box').css('height', wh - th);
		var lefth = (wh - th - 20) / 2;
		$('#left .chartBox').css('height', lefth);
		$('#mapcharts').css('height', lefth);
		var middleh = wh - th - 10;
		// console.log(middleh)
		$('#middle .chartBox').css('height', middleh);
		var righth = (wh - th - 30) / 3;
		$('#right .chartBox').css('height', righth);
		$('#Pcp').css('height', $('#Pcp').parent().height() - $('#Pcp').parent().find('div.title').height())
		$('#Tcopt').css('height', $('#Tcopt').parent().height() - $('#Tcopt').parent().find('div.title')
		.height())
		$('#GHmap').css('height', $('#GHmap').parent().height() - $('#GHmap').parent().find('div.title')
		.height())
		$('#Ets').css('height', $('#Ets').parent().height() - $('#Ets').parent().find('div.title').height())
		$('#Qcst').css('height', $('#Qcst').parent().height() - $('#Qcst').parent().find('div.title').height())
		$('#Cc').css('height', $('#Cc').parent().height() - $('#Cc').parent().find('div.title').height())

		// 这个是中间模型的
		sdsdata.dataGHmap();

		// CurrentArea = "GUIZHOU";   //修改all为具体的某个地址GUIZHOU
		CurrentArea = "GUANGDONG";   //修改all为具体的某个地址GUIZHOU
		GetData(CurrentArea);
	},
	// dataEts: function dataEts(res) {
	// 	if (EtsChart != null && EtsChart != "" && EtsChart != undefined) {
	// 		EtsChart.dispose(); //销毁
	// 	}
	// 	EtsChart = echarts.init(document.getElementById('Ets'));
	// 	var dataname = [];
	// 	var datavalue = [];
	// 	for (i = 0; i < res.data.length; i++) {
	// 		dataname.push(res.data[i].name);
	// 		datavalue.push(res.data[i].value);
	// 	}
	// 	// 指定图表的配置项和数据
	// 	var option = {
	// 		tooltip: {},
	// 		legend: {
	// 			// top: 10,
	// 			right: 10,
	// 			textStyle: {
	// 				color: '#fff'
	// 			},
	// 			data: ['Blood pressure parameter']
	// 		},
	// 		grid: {
	// 			left: 10,
	// 			top: 45,
	// 			right: 15,
	// 			bottom: 5,
	// 			containLabel: true
	// 		},
	// 		xAxis: {
	// 			axisLine: {
	// 				show: true,
	// 				lineStyle: {
	// 					color: '#fff'
	// 				}
	// 			},
	// 			axisLabel: {
	// 				color: '#c9e5ff',
	// 				interval: 'auto',
	// 				fontSize: 11,
	// 				nameRotate: 45,
	// 			},
	// 			axisTick: {
	// 				show: false,
	// 			},
	// 			axisLine: {
	// 				show: false,
	// 			},
	// 			data: dataname
	// 		},
	// 		yAxis: {
	// 			axisLine: {
	// 				// show: false,
	// 				lineStyle: {
	// 					color: '#fff'
	// 				}
	// 			},
	// 			axisLabel: {
	// 				color: '#c9e5ff'
	// 			},
	// 			axisTick: {
	// 				show: false,
	// 			},
	// 			splitLine: {
	// 				// show: false,   网格线
	// 				lineStyle: {
	// 					color: ['#124051']
	// 				}
	// 			}
	// 		},
	// 		series: [{
	// 			name: 'Blood pressure parameter',
	// 			type: 'bar',
	// 			barWidth: 20,
	// 			label: {
	// 				// show: true,
	// 				color: '#fff',
	// 				position: 'top'
	// 			},
	// 			itemStyle: {
	// 				color: new echarts.graphic.LinearGradient(
	// 					0, 0, 0, 1,
	// 					[{
	// 							offset: 0,
	// 							color: '#00e5f3'
	// 						},
	// 						{
	// 							offset: 0.5,
	// 							color: '#1464c6'
	// 						},
	// 						{
	// 							offset: 1,
	// 							color: '#600f95'
	// 						}
	// 					]
	// 				)
	// 			},
	// 			data: datavalue
	// 		}]
	// 	};

	// 	// 使用刚指定的配置项和数据显示图表。
	// 	EtsChart.setOption(option); //右边图表1
	// },
	dataEts: function dataEts(res) {
		if (EtsChart != null && EtsChart != "" && EtsChart != undefined) {
			EtsChart.dispose(); //销毁
		}
		EtsChart = echarts.init(document.getElementById('Ets'));
		var dataname = [];
		var datavalue = [];
		for (i = 0; i < res.data.length; i++) {
			dataname.push(res.data[i].name);
			datavalue.push(res.data[i].value);
		}
		// 指定图表的配置项和数据
		var option = {
			tooltip: {},
			legend: {
				// top: 10,
				right: 10,
				textStyle: {
					color: '#fff'
				},
				data: ['Blood pressure parameter']
			},
			grid: {
				left: 10,
				top: 45,
				right: 15,
				bottom: 5,
				containLabel: true
			},
			xAxis: {
				axisLine: {
					show: true,
					lineStyle: {
						color: '#fff'
					}
				},
				axisLabel: {
					color: '#c9e5ff',
					interval: 'auto',
					fontSize: 11,
					nameRotate: 45,
					// formatter:function(value)
					// {
					// 	var ret = "";//拼接加\n返回的类目项  
					// 	var maxLength = 3;//每项显示文字个数  
					// 	var valLength = value.length;//X轴类目项的文字个数  
					// 	var rowN = Math.ceil(valLength / maxLength); //类目项需要换行的行数  
					// 	if (rowN > 1)//如果类目项的文字大于3,  
					// 	{  
					// 		for (var i = 0; i < rowN; i++) {  
					// 			var temp = "";//每次截取的字符串  
					// 			var start = i * maxLength;//开始截取的位置  
					// 			var end = start + maxLength;//结束截取的位置   
					// 			temp = value.substring(start, end) + "\n";  
					// 			ret += temp; //最终的字符串  
					// 		}  
					// 		return ret;  
					// 	} else {return value;}
					// }
				},
				axisTick: {
					show: false
				},
				axisLine: {
					show: false
				},
				data: dataname
			},
			yAxis: {
				axisLine: {
					show: false,
					lineStyle: {
						color: '#fff'
					}
				},
				axisLabel: {
					color: '#c9e5ff'
				},
				axisTick: {
					show: false
				},
				splitLine: {
					lineStyle: {
						color: ['#124051']
					}
				}
			},
			series: [{
				name: 'Blood pressure parameter',
				type: 'bar',
				barWidth: 25,
				itemStyle: {
					color: new echarts.graphic.LinearGradient(
						0, 0, 0, 1,
						[{
								offset: 0,
								color: '#00e5f3'
							},
							{
								offset: 1,
								color: '#1464c6'
							}
						]
					)
				},
				data: datavalue
			}, {
				name: '',
				type: 'line',
				barWidth: 20,
				label: {
					// show: true,
					color: '#fff',
					position: 'top'
				},
				itemStyle: {
					color: new echarts.graphic.LinearGradient(
						0, 0, 0, 1,
						[{
								offset: 0,
								color: '#e489ea'
							},
							{
								offset: 0.5,
								color: '#fb65e2'
							},
							{
								offset: 1,
								color: '#b541bf'
							}
						]
					)
				},
				data: datavalue
			}]
		};
		// 使用刚指定的配置项和数据显示图表。
		EtsChart.setOption(option);
	},
	// 将该部分更改为健康咨询
	// dataTcopt: function dataTcopt(res) {
	// 	if (TcoptChart != null && TcoptChart != "" && TcoptChart != undefined) {
	// 		TcoptChart.dispose(); //销毁
	// 	}
	// 	TcoptChart = echarts.init(document.getElementById('Tcopt'));
	// 	var dataname = [];
	// 	var datavalue = [];
	// 	for (i = 0; i < res.data.length; i++) {
	// 		dataname.push(res.data[i].name);
	// 		datavalue.push(res.data[i].value);
	// 	}
	// 	// 指定图表的配置项和数据
	// 	var option = {
	// 		tooltip: {
	// 			formatter: '{b} : {c} ({d}%)'
	// 		},
	// 		color: ["#5B90FA", "#5AD9A6", "#7289b1", "#F6BD16", "#E86949", "#6EC8EB", "#926FC9", "#FF9D4C",
	// 			"#269A99", "#FF98C3"
	// 		],
	// 		series: [{
	// 			name: 'Total composition of product types',
	// 			radius: '50%',
	// 			center: ['50%', '50%'],
	// 			left: 0,
	// 			top: 0,
	// 			bottom: 0,
	// 			type: 'pie',
	// 			barWidth: 10,
	// 			label: {
	// 				color: 'auto',
	// 				position: 'outer',
	// 				alignTo: 'edge',
	// 				margin: 20,
	// 				bleedMargin: 5,
	// 				formatter: '{b}:{c}'
	// 			},
	// 			itemStyle: {},
	// 			data: res.data
	// 		}, {
	// 			name: 'Total composition of product types',
	// 			radius: '50%',
	// 			center: ['50%', '50%'],
	// 			left: 0,
	// 			top: 0,
	// 			bottom: 0,
	// 			type: 'pie',
	// 			barWidth: 10,
	// 			label: {
	// 				color: '#ffffff',
	// 				position: 'inside',
	// 				margin: 20,
	// 				bleedMargin: 5,
	// 				formatter: '{d}%'
	// 			},
	// 			data: res.data
	// 		}]
	// 	};
	// 	// 使用刚指定的配置项和数据显示图表。
	// 	TcoptChart.setOption(option);  //左边图2
	// },
	// dataPcp: function dataPcp(res) {
	// 	if (PcpChart != null && PcpChart != "" && PcpChart != undefined) {
	// 		PcpChart.dispose(); //销毁
	// 	}
	// 	PcpChart = echarts.init(document.getElementById('Pcp'));
	// 	var dataname = [];
	// 	var datavalue = [];
	// 	for (i = 0; i < res.data.length; i++) {
	// 		dataname.push(res.data[i].name);
	// 		datavalue.push(res.data[i].value);
	// 	}
	// 	// 指定图表的配置项和数据
	// 	var option = {
	// 		color: ["#1890FF", "#FF2D55", "#fcfcff", "#25fff8", "#e2ff21", "#ff6822"],
	// 		series: [{
	// 			name: 'Product classification proportion',
	// 			type: 'pie',
	// 			radius: ['45%', '60%'],
	// 			center: ['50%', '38%'],
	// 			avoidLabelOverlap: false,
	// 			hoverAnimation: false,
	// 			silent: true,
	// 			data: res.data
	// 		}]
	// 	};
	// 	var optionData = option.series[0].data;
	// 	var optionDataAll = 0;
	// 	var optionDataLv = 0;
	// 	for (i = 0; i < optionData.length; i++) {
	// 		optionDataAll += optionData[i].value;
	// 	}
	// 	optionDataLv = optionDataAll / 100;
	// 	option.legend = {
	// 		type: 'scroll',
	// 		orient: 'vertical',
	// 		scrollDataIndex: 1,
	// 		top: '75%',
	// 		left: '10%',
	// 		textStyle: {
	// 			color: "#fff"
	// 		},
	// 		pageButtonPosition: 'end',
	// 		pageIconColor: '#fff',
	// 		pageIconInactiveColor: '#002139',
	// 		pageTextStyle: {
	// 			color: "#fff"
	// 		},
	// 		formatter(params) {
	// 			for (i = 0; i < optionData.length; i++) {
	// 				if (optionData[i].name == params) {
	// 					return params + " : " + (optionData[i].value / optionDataLv).toFixed(2) + "% " +
	// 						"(" + optionData[i].value + ")";
	// 				}
	// 			}
	// 		}
	// 	};
	// 	option.series[0].label = {
	// 		normal: {
	// 			color: '#fff',
	// 			position: 'center',
	// 			fontSize: 16,
	// 			formatter: ['Total', '', 'products', '', optionDataAll].join('\n'),
	// 			rich: {
	// 				value: {
	// 					color: '#ffffff',
	// 					fontSize: 40,
	// 					fontWeight: 'bold',
	// 					lineHeight: 40,
	// 				},
	// 				name: {
	// 					color: '#909399',
	// 					lineHeight: 20
	// 				},
	// 			},
	// 		},
	// 	}
	// 	// 使用刚指定的配置项和数据显示图表。
	// 	PcpChart.setOption(option);  //
	// },
	dataQcst: function dataQcst(res) {
		if (QcstChart != null && QcstChart != "" && QcstChart != undefined) {
			QcstChart.dispose(); //销毁
		}
		QcstChart = echarts.init(document.getElementById('Qcst'));
		var dataname = [];
		var datavalue = [];
		for (i = 0; i < res.data.length; i++) {
			dataname.push(res.data[i].name);
			datavalue.push(res.data[i].value);
		}
		// 指定图表的配置项和数据
		var option = {
			tooltip: {},
			legend: {
				// top: 10,
				right: 10,
				textStyle: {
					color: '#fff'
				},
				data: ['Blood oxygen parameter']
			},
			grid: {
				left: 10,
				top: 45,
				right: 15,
				bottom: 5,
				containLabel: true
			},
			xAxis: {
				axisLine: {
					show: true,
					lineStyle: {
						color: '#fff'
					}
				},
				axisLabel: {
					color: '#c9e5ff',
					interval: 'auto',
					fontSize: 11,
					nameRotate: 45,
					// formatter:function(value)
					// {
					// 	var ret = "";//拼接加\n返回的类目项  
					// 	var maxLength = 3;//每项显示文字个数  
					// 	var valLength = value.length;//X轴类目项的文字个数  
					// 	var rowN = Math.ceil(valLength / maxLength); //类目项需要换行的行数  
					// 	if (rowN > 1)//如果类目项的文字大于3,  
					// 	{  
					// 		for (var i = 0; i < rowN; i++) {  
					// 			var temp = "";//每次截取的字符串  
					// 			var start = i * maxLength;//开始截取的位置  
					// 			var end = start + maxLength;//结束截取的位置   
					// 			temp = value.substring(start, end) + "\n";  
					// 			ret += temp; //最终的字符串  
					// 		}  
					// 		return ret;  
					// 	} else {return value;}
					// }
				},
				axisTick: {
					show: false
				},
				axisLine: {
					show: false
				},
				data: dataname
			},
			yAxis: {
				axisLine: {
					show: false,
					lineStyle: {
						color: '#fff'
					}
				},
				axisLabel: {
					color: '#c9e5ff'
				},
				axisTick: {
					show: false
				},
				splitLine: {
					lineStyle: {
						color: ['#124051']
					}
				}
			},
			series: [{
				name: 'Blood oxygen parameter',
				type: 'bar',
				barWidth: 25,
				itemStyle: {
					color: new echarts.graphic.LinearGradient(
						0, 0, 0, 1,
						[{
								offset: 0,
								color: '#00e5f3'
							},
							{
								offset: 1,
								color: '#1464c6'
							}
						]
					)
				},
				data: datavalue
			}, {
				name: '',
				type: 'line',
				barWidth: 20,
				label: {
					// show: true,
					color: '#fff',
					position: 'top'
				},
				itemStyle: {
					color: new echarts.graphic.LinearGradient(
						0, 0, 0, 1,
						[{
								offset: 0,
								color: '#e489ea'
							},
							{
								offset: 0.5,
								color: '#fb65e2'
							},
							{
								offset: 1,
								color: '#b541bf'
							}
						]
					)
				},
				data: datavalue
			}]
		};
		// 使用刚指定的配置项和数据显示图表。
		QcstChart.setOption(option);
	},
	dataCc: function dataCc(res) {
		if (CcChart != null && CcChart != "" && CcChart != undefined) {
			CcChart.dispose(); //销毁
		}
		CcChart = echarts.init(document.getElementById('Cc'));
		var dataname = [];
		var datavalue = [];
		for (i = 0; i < res.data.length; i++) {
			dataname.push(res.data[i].name);
			datavalue.push(res.data[i].value);
		}

		// 指定图表的配置项和数据
		var option = {
			tooltip: {},
			legend: {
				// top: 10,
				right: 10,
				textStyle: {
					color: '#fff'
				},
				data: ['Heart rate parameter']
			},
			grid: {
				left: 10,
				top: 45,
				right: 15,
				bottom: 5,
				containLabel: true
			},
			xAxis: {
				axisLine: {
					show: true,
					lineStyle: {
						color: '#fff'
					}
				},
				axisLabel: {
					color: '#c9e5ff',
					interval: 'auto',
					fontSize: 11,
					nameRotate: 45,
					// formatter:function(value)
					// {
					// 	var ret = "";//拼接加\n返回的类目项  
					// 	var maxLength = 3;//每项显示文字个数  
					// 	var valLength = value.length;//X轴类目项的文字个数  
					// 	var rowN = Math.ceil(valLength / maxLength); //类目项需要换行的行数  
					// 	if (rowN > 1)//如果类目项的文字大于3,  
					// 	{  
					// 		for (var i = 0; i < rowN; i++) {  
					// 			var temp = "";//每次截取的字符串  
					// 			var start = i * maxLength;//开始截取的位置  
					// 			var end = start + maxLength;//结束截取的位置   
					// 			temp = value.substring(start, end) + "\n";  
					// 			ret += temp; //最终的字符串  
					// 		}  
					// 		return ret;  
					// 	}  
					// 	else {  
					// 		return value;
					// 	}
					// }
				},
				axisTick: {
					show: false,
				},
				axisLine: {
					show: false,
				},
				data: dataname
			},
			yAxis: {
				axisLine: {
					show: false,
					lineStyle: {
						color: '#fff'
					}
				},
				axisLabel: {
					color: '#c9e5ff'
				},
				axisTick: {
					show: false,
				},
				splitLine: {
					// show: false,
					lineStyle: {
						color: ['#124051']
					}
				}
			},
			series: [{
				name: 'Heart rate parameter',
				type: 'bar',
				barWidth: 25,
				itemStyle: {
					color: new echarts.graphic.LinearGradient(
						0, 0, 0, 1,
						[{
								offset: 0,
								color: '#1b5fc6'
							},
							{
								offset: 0.5,
								color: '#442b95'
							},
							{
								offset: 1,
								color: '#600f95'
							}
						]
					)
				},
				data: datavalue
			}, {
				name: '',
				type: 'line',
				barWidth: 20,
				label: {
					// show: true, 显示具体参数值
					color: '#fff',
					position: 'top'
				},
				itemStyle: {
					color: new echarts.graphic.LinearGradient(
						0, 0, 0, 1,
						[{
								offset: 0,
								color: '#f4ccfb'
							},
							{
								offset: 0.5,
								color: '#fb65e2'
							},
							{
								offset: 1,
								color: '#b541bf'
							}
						]
					)
				},
				data: datavalue
			}]
		};
		// 使用刚指定的配置项和数据显示图表。
		CcChart.setOption(option);
	},
	dataGHmap: function dataGHmap() {
		window.digit_glb.load_vrJson({
			"rotate": {
				"x": 0,
				"y": 0,
				"z": -(90 * Math.PI) / 180
			},
			"size": 0.6,
			"data": {
				"url": "./public/fbx_output_3.glb",
				"animation": 0,
				"animationSpeed": 0.03
			}
		})

		setTimeout(function() {
			window.digit_glb.upWH()
		}, 1000)
	}
};

function getTime() {
	var dd = new Date(); //获取当前时间对象
	var year = dd.getFullYear(); //年份
	var month = dd.getMonth() + 1; //月份+1得到当前月
	var day = dd.getDate(); //获取日
	var ww = dd.getDay(); //获取星期
	//alert(ww);
	var weeks = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
	var week = weeks[ww]; //根据星期，获取当前星期几;
	var hour = dd.getHours();
	var minutes = dd.getMinutes();
	var second = dd.getSeconds();
	if (month < 10) {
		month = "0" + month; //判断小于10，前面+0填充
	}
	if (day < 10) {
		day = "0" + day;
	}
	if (hour < 10) {
		hour = "0" + hour;
	}
	if (minutes < 10) {
		minutes = "0" + minutes;
	}
	if (second < 10) {
		second = "0" + second;
	}
	var html = `<div style="padding: 5px;">
				<h1>` + hour + ':' + minutes + ':' + second + `</h1>
			</div>
			<div style="padding: 8px 5px;">
				<p>` + month + '/' + day + '/' + year + `</p>
				<p>` + week + `</p>
			</div>`;
	$("#setTime").html(html);
	setTimeout(getTime, 1000);
}

function AddressH() {
	$("#address").css('line-height', '65.5px');
	var html = `&#xe606;&nbsp;` + CurrentArea;
	$("#address").html(html);
	var addressht = $("#address").height()
	if (addressht > 120) {
		if (addressht > 190) {
			$("#address").css('line-height', 65 / 3 + "px");
		} else {
			$("#address").css('line-height', 65 / 2 + "px");
		}
	}
}

$(function() {
	getTime();
	sdsdata.init();
	$(window).resize(function() {
		sdsdata.init();
	});

})

function GetData(region) {
	AddressH();
	// sdsdata.dataEts();
	// sdsdata.dataTcopt();
	// sdsdata.dataPcp();
	// sdsdata.dataQcst();
	// sdsdata.dataCc();
	// sdsdata.dataGHmap();

	$.ajax({
		url: './json/json1/Eoe.json',
		type: "get",
		data: {
			"region": region
		},
		success: function(res) {
			$("#Noe").text(res.enterprises);
			$("#Yoyg").text(res.growth);
		}
	})
	$.ajax({
		url: './json/json1/Ets.json',
		type: "get",
		data: {
			"region": region
		},
		success: function(res) {
			sdsdata.dataEts(res);
		}
	})
	$.ajax({
		url: './json/json1/Tcopt.json',
		type: "get",
		data: {
			"region": region
		},
		success: function(res) {
			sdsdata.dataTcopt(res);
		}
	})
	$.ajax({
		url: './json/json1/Pcp.json',
		type: "get",
		data: {
			"region": region
		},
		success: function(res) {
			sdsdata.dataPcp(res);
		}
	})
	$.ajax({
		url: './json/json1/Qcst.json',
		type: "get",
		data: {
			"region": region
		},
		success: function(res) {
			sdsdata.dataQcst(res);
		}
	})
	$.ajax({
		url: './json/json1/Cc.json',
		type: "get",
		data: {
			"region": region
		},
		success: function(res) {
			sdsdata.dataCc(res);
		}
	})
	$.ajax({
		url: './json/json1/change.json',
		type: "get",
		data: {
			"region": region
		},
		success: function(res) {
			$("#No").text(res.state);
			$("#Yo").text(res.posture);
		}
	})
	
}
