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
		var lefth1 = (wh - th - 20)/9*5;
		var lefth2 = (wh - th - 20)/9*4;
		$('#left .chartBox').css('height', lefth2);
		$('#mapcharts').css('height', lefth1);
		var middleh = (wh - th - 20)/2;
		$('#middle .chartBox').css('height', middleh);
		var righth = (wh - th - 20)/2;
		$('#right .chartBox').css('height', righth);
		$.ajax({
			url: '../public/map/ghana.json',
			type: "get",
			success: function(res){
				sdsdata.dataGHmap(res);
			}
		});
		CurrentArea = "All";
		GetData(CurrentArea);
	},
	dataEts: function dataEts(res) {
		if (EtsChart != null && EtsChart != "" && EtsChart != undefined) {
			EtsChart.dispose(); //销毁
		}
		EtsChart = echarts.init(document.getElementById('Ets'));
		var dataname = [];
		var datavalue = [];
		for(i=0;i<res.data.length;i++){
			dataname.push(res.data[i].name);
			datavalue.push(res.data[i].value);
		}
		// 指定图表的配置项和数据
		var option = {
			tooltip: {},
			legend: {
				top: 10,
				right: 10,
				textStyle: {
					color: '#fff'
				},
				data: ['Number of enterprises']
			},
			grid: {
				left: 10,
				top: 50,
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
				name: 'Number of enterprises',
				type: 'bar',
				barWidth : 20,
				label: {
					show: true,
					color: '#fff',
					position: 'top'
				},
				itemStyle: {
					color: new echarts.graphic.LinearGradient(
						0, 0, 0, 1,
						[{
								offset: 0,
								color: '#00e5f3'
							},
							{
								offset: 0.5,
								color: '#1464c6'
							},
							{
								offset: 1,
								color: '#600f95'
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
	dataTcopt: function dataTcopt(res) {
		if (TcoptChart != null && TcoptChart != "" && TcoptChart != undefined) {
			TcoptChart.dispose(); //销毁
		}
		TcoptChart = echarts.init(document.getElementById('Tcopt'));
		var dataname = [];
		var datavalue = [];
		for(i=0;i<res.data.length;i++){
			dataname.push(res.data[i].name);
			datavalue.push(res.data[i].value);
		}
		// 指定图表的配置项和数据
		var option = {
			tooltip: {
				formatter: '{b} : {c} ({d}%)'
			},
			color: ["#5B90FA","#5AD9A6","#7289b1","#F6BD16","#E86949","#6EC8EB","#926FC9","#FF9D4C","#269A99","#FF98C3"],
			series: [{
				name: 'Total composition of product types',
				radius: '50%',
				center: ['50%', '50%'],
				left: 0,
				top: 0,
				bottom: 0,
				type: 'pie',
				barWidth : 10,
				label: {
					color: 'auto',
					position: 'outer',
					alignTo: 'edge',
					margin: 20,
					bleedMargin: 5,
					formatter:'{b}:{c}'
				},
				itemStyle: {},
				data: res.data
			},{
				name: 'Total composition of product types',
				radius: '50%',
				center: ['50%', '50%'],
				left: 0,
				top: 0,
				bottom: 0,
				type: 'pie',
				barWidth : 10,
				label: {
					color: '#ffffff',
					position: 'inside',
					margin: 20,
					bleedMargin: 5,
					formatter:'{d}%'
				},
				data: res.data
			}]
		};
		// 使用刚指定的配置项和数据显示图表。
		TcoptChart.setOption(option);
	},
	dataPcp: function dataPcp(res) {
		if (PcpChart != null && PcpChart != "" && PcpChart != undefined) {
			PcpChart.dispose(); //销毁
		}
		PcpChart = echarts.init(document.getElementById('Pcp'));
		var dataname = [];
		var datavalue = [];
		for(i=0;i<res.data.length;i++){
			dataname.push(res.data[i].name);
			datavalue.push(res.data[i].value);
		}
		// 指定图表的配置项和数据
		var option = {
			color: ["#1890FF","#FF2D55","#fcfcff","#25fff8","#e2ff21","#ff6822"],
			series: [{
				name: 'Product classification proportion',
				type: 'pie',
				radius: ['45%', '60%'],
				center: ['50%','38%'],
				avoidLabelOverlap: false,
				hoverAnimation: false,
				silent: true,
				data: res.data
			}]
		};
		var optionData = option.series[0].data;
		var optionDataAll = 0;
		var optionDataLv = 0;
		for(i=0;i<optionData.length;i++){
			optionDataAll +=  optionData[i].value;
		}
		optionDataLv = optionDataAll/100;
		option.legend = {
			type: 'scroll',
			orient: 'vertical',
			scrollDataIndex: 1,
			top: '75%',
			// bottom: '5%',
			left: '10%',
			textStyle: {color: "#fff"},
			pageButtonPosition: 'end',
			pageIconColor: '#fff',
			pageIconInactiveColor: '#002139',
			pageTextStyle: {color: "#fff"},
			formatter(params) {
				for(i=0;i<optionData.length;i++){
					if(optionData[i].name == params){
						return params+" : "+(optionData[i].value/optionDataLv).toFixed(2)+"% "+"("+optionData[i].value+")";
					}
				}
			}
		};
		option.series[0].label = {
			normal: {
				color: '#fff',
				position: 'center',
				fontSize: 16,
				formatter: ['Total','','products','',optionDataAll].join('\n'),
				rich: {
					value: {
						color: '#ffffff',
						fontSize: 40,
						fontWeight: 'bold',
						lineHeight: 40,
					},
					name: {
						color: '#909399',
						lineHeight: 20
					},
				},
			},
		}
		// 使用刚指定的配置项和数据显示图表。
		PcpChart.setOption(option);
	},
	dataQcst: function dataQcst(res) {
		if (QcstChart != null && QcstChart != "" && QcstChart != undefined) {
			QcstChart.dispose(); //销毁
		}
		QcstChart = echarts.init(document.getElementById('Qcst'));
		var dataname = [];
		var datavalue = [];
		for(i=0;i<res.data.length;i++){
			dataname.push(res.data[i].name);
			datavalue.push(res.data[i].value);
		}
		// 指定图表的配置项和数据
		var option = {
			tooltip: {},
			legend: {
				top: 10,
				right: 10,
				textStyle: {color: '#fff'},
				data: ['Number of times']
			},
			grid: {
				left: 10,
				top: 50,
				right: 15,
				bottom: 5,
				containLabel: true
			},
			xAxis: {
				axisLine: {
					show: true,
					lineStyle: {color: '#fff'}
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
				axisTick: {show: false},
				axisLine: {show: false},
				data: dataname
			},
			yAxis: {
				axisLine: {
					show: false,
					lineStyle: {color: '#fff'}
				},
				axisLabel: {color: '#c9e5ff'},
				axisTick: {show: false},
				splitLine: {lineStyle: {color: ['#124051']}}
			},
			series: [{
				name: 'Number of times',
				type: 'bar',
				barWidth : 25,
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
			},{
				name: '',
				type: 'line',
				barWidth : 20,
				label: {
					show: true,
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
						}]
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
		for(i=0;i<res.data.length;i++){
			dataname.push(res.data[i].name);
			datavalue.push(res.data[i].value);
		}
		// 指定图表的配置项和数据
		var option = {
			tooltip: {},
			legend: {
				top: 10,
				right: 10,
				textStyle: {
					color: '#fff'
				},
				data: ['Number of complaints']
			},
			grid: {
				left: 10,
				top: 50,
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
				name: 'Number of complaints',
				type: 'bar',
				barWidth : 25,
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
						}]
					)
				},
				data: datavalue
			},{
				name: '',
				type: 'line',
				barWidth : 20,
				label: {
					show: true,
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
						}]
					)
				},
				data: datavalue
			}]
		};
		// 使用刚指定的配置项和数据显示图表。
		CcChart.setOption(option);
	},
	dataGHmap: function dataGHmap(res){
		var mapdata = [];
		var resdata = res.features;
		for(var i=0;i<resdata.length;i++){
			mapdata.push({
				"name": resdata[i].properties.name,
				"value": 0
			})
		}
		
		if (GHmapChart != null && GHmapChart != "" && GHmapChart != undefined) {
			GHmapChart.dispose(); //销毁
		}
		echarts.registerMap('world', {geoJSON: res});
		GHmapChart = echarts.init(document.getElementById('GHmap'));
		var option = {
			// visualMap: {
			// 	show: false,
			// 	left: 'right',
			// 	categories: ['1'],
			// 	inRange: {
			// 		color: ['#f46d43']
			// 	},
			// 	text:['High','Low'],  // 文本，默认为数值文本
			// 	calculable: true
			// },
			series: {
				type: 'map',
				map: 'world',
				roam: true,
				data: mapdata,
				select: {
					label: {
						color: '#fff'
					},
					itemStyle: {
						areaColor: '#f46d43'
					}
				},
				itemStyle: {
					normal: {
						borderWidth: 1,
						borderColor: '#fff',
						areaColor: '#2a83b3'
					},
					emphasis: {
						show: true,
						areaColor: '#3066ba',//鼠标滑过区域颜色
						label: {
							show: true,
							textStyle: {
								color: '#fff'
							}
						}
					}   
				},
			}
		};
		GHmapChart.setOption(option);
		GHmapChart.on('click', function (params) {
			for(var i=0;i<mapdata.length;i++){
				if(params.name == mapdata[i].name){
					if(mapdata[i].value == "1"){
						mapdata[i].value="0";
					}else{mapdata[i].value="1";}
				}else{mapdata[i].value="0";}
			}
			// GHmapChart.setOption(option);
			for(var i=0;i<mapdata.length;i++){
				if(mapdata[i].value == "1"){
					CurrentArea = mapdata[i].name;
					break;
				}else{
					CurrentArea = "All";
				}
			}
			GetData(CurrentArea);
		});
	}
};
function getTime(){
	var dd=new Date();//获取当前时间对象
	var year=dd.getFullYear();//年份
	var month=dd.getMonth()+1;//月份+1得到当前月
	var day=dd.getDate();//获取日
	var ww=dd.getDay();//获取星期
	//alert(ww);
	var weeks=["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"];
	var week=weeks[ww];//根据星期，获取当前星期几;
	var hour=dd.getHours();
	var minutes=dd.getMinutes();
	var second=dd.getSeconds();
	if(month<10){
		month = "0"+month;//判断小于10，前面+0填充
	}
	if(day<10){
		day = "0"+day;
	}
	if(hour<10){
		hour = "0"+hour;
	}
	if(minutes<10){
		minutes = "0"+minutes;
	}
	if(second<10){
		second = "0"+second;
	}
	var html = `<div style="padding: 5px;">
				<h1>`+hour+':'+minutes+':'+second+`</h1>
			</div>
			<div style="padding: 8px 5px;">
				<p>`+month+'/'+day+'/'+year+`</p>
				<p>`+week+`</p>
			</div>`;
	$("#setTime").html(html);
	setTimeout(getTime,1000);
}
$(function(){
	getTime();
	sdsdata.init();
	$(window).resize(function() {
		sdsdata.init();
	});
	
})

function GetData(region){
	var html = `&#xe606;&nbsp;`+CurrentArea;
	$("#address").html(html);
	// sdsdata.dataEts();
	// sdsdata.dataTcopt();
	// sdsdata.dataPcp();
	// sdsdata.dataQcst();
	// sdsdata.dataCc();
	// sdsdata.dataGHmap();
	
	
	$.ajax({
		url: '../json/json1/Ets.json',
		type: "get",
		data: {"region": region},
		success: function(res){
			sdsdata.dataEts(res);
		}
	})
	$.ajax({
		url: '../json/json1/Tcopt.json',
		type: "get",
		data: {"region": region},
		success: function(res){
			sdsdata.dataTcopt(res);
		}
	})
	$.ajax({
		url: '../json/json1/Pcp.json',
		type: "get",
		data: {"region": region},
		success: function(res){
			sdsdata.dataPcp(res);
		}
	})
	$.ajax({
		url: '../json/json1/Qcst.json',
		type: "get",
		data: {"region": region},
		success: function(res){
			sdsdata.dataQcst(res);
		}
	})
	$.ajax({
		url: '../json/json1/Cc.json',
		type: "get",
		data: {"region": region},
		success: function(res){
			sdsdata.dataCc(res);
		}
	})
}