(function($){
    var clean_display = function($dp) {
	$dp.text("");
    }
    var display_original_image = function(data) {
	clean_display($dp);
	var $dt = $("<div id=\"dt\">");
	$.each(data.origimages, function(index, value) {
	    $.each(value, function(key, value) {
		key!="value" && $("<p>").text(key+":"+value).appendTo($dt);
	    });
	    
	    $.each(value.split_list, function(index, value) {
		$.getJSON("/v1/splited_images/"+value+"/character", function(data) {
		    $("#spval"+index).text(data.character);
		});
	    });
	    spans = []
	    $p = $("<p>").append($("<span>").text("value:"));
	    for(var i=0; i < 4; i++) {
		$p.append($("<span>").attr("id", "spval"+i).text("*"));
	    }
	    $p.append($("<input>").attr("id", "dpivs")).append($("<button>").attr("id", "dpbtn").text("contribute")).appendTo($dt);
	    $("<input>").attr("id", "dphoid").css("display", "none").val(value.id).appendTo($dt);
	    $("<p>").append($("<img>").attr("src", value.path)).appendTo($dt);
	});
	$dt.appendTo($dp);
    }
    var $dp = $("#dp");
    var $oq = $("#oq");
    $("#oqb").click(function(evt){
	$.getJSON("/v1/original_images/" + $oq.val(), display_original_image);
    });
    $("#sqb").click(function(evt){
	console.log("sqb press");
    });
    $(document).on("click", "#dpbtn", function(evt) {
	$ivs = $("#dpivs");
	$oid = $("#dphoid");

	code = $ivs.val();
	if (code != 4) {
	    $.getJSON("/v1/original_images/" + $oq.val(), display_original_image);
	}
	$.post("/v1/original_images/"+$oid.val()+"/value/"+code,function(data){
	    $.getJSON("/v1/original_images/" + $oq.val(), display_original_image);
	});
    });
})($);