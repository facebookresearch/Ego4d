(this.webpackJsonpreview=this.webpackJsonpreview||[]).push([[0],{122:function(e,t,i){},148:function(e,t,i){},151:function(e,t,i){},152:function(e,t,i){},156:function(e,t,i){},175:function(e,t,i){},176:function(e,t,i){"use strict";i.r(t);var n=i(0),r=i.n(n),a=i(11),s=i.n(a),c=i(18),o=i(8),l=i(9),d=i(29),j=i(179),h=i(27),b=i(20),m=(i(122),i(1));function u(e){var t=e.item,i=Object(n.useState)(0),r=Object(l.a)(i,2),a=r[0],s=r[1],c=Object(n.useRef)();Object(n.useEffect)((function(){s(c.current.offsetWidth)}),[]);var o=a<1e3;return Object(m.jsx)("div",{ref:c,className:"json-item-renderer",id:"item-view-".concat(t&&t.id),children:Object(m.jsxs)(j.c,{elevation:b.a.TWO,interactive:o,className:"json-item-card",children:[Object(m.jsx)("pre",{className:o?"json-item-renderer-pre small":"json-item-renderer-pre",children:JSON.stringify(t&&t.data)}),Object(m.jsx)(j.e,{children:Object(m.jsxs)("b",{children:["ID: ",t&&t.id]})})]})})}i(148);function O(e){var t=e.items,i=e.itemRenderer,n=void 0===i?u:i;return t&&t.length>0?Object(m.jsx)("div",{className:"default-collection-renderer-container",children:t.map((function(e){return Object(m.jsx)(c.b,{to:"/".concat(e.id),style:{textDecoration:"none"},id:"item-".concat(e.id),children:Object(m.jsx)(n,{item:e})},e.id)}))}):null}var v=i(19),p=(i(57),function(e){var t=arguments.length>1&&void 0!==arguments[1]?arguments[1]:[],i={},n=e.split(" ").filter((function(i){if(!i||""===i||1===i.length)return!1;if(!isNaN(parseFloat(e)))return!1;var n,r=Object(v.a)(t);try{for(r.s();!(n=r.n()).done;){var a=n.value;if(i.toLowerCase().indexOf(a.toLowerCase())>-1)return!1}}catch(s){r.e(s)}finally{r.f()}return!0}));n=n.map((function(e){return e.replace(/[.,/#!$%^&*;:{}=\-_`~()]/g,"")}));var r,a=Object(v.a)(n);try{for(a.s();!(r=a.n()).done;){var s=r.value;s in i?i[s]+=1:i[s]=1}}catch(c){a.e(c)}finally{a.f()}return i}),f=function e(t){var i,n=arguments.length>1&&void 0!==arguments[1]?arguments[1]:[],r=arguments.length>2&&void 0!==arguments[2]?arguments[2]:[],a={},s=Object.keys(t).filter((function(e){var t,i=Object(v.a)(r);try{for(i.s();!(t=i.n()).done;){if(e===t.value)return!1}}catch(n){i.e(n)}finally{i.f()}return!0})),c=Object(v.a)(s);try{for(c.s();!(i=c.n()).done;){var o=i.value,l=t[o];if(l&&""!==l)if("object"===typeof l)for(var d=e(l,n,r),j=Object.keys(d),h=0,b=j;h<b.length;h++){var m=b[h];a[m]=m in a?d[m]+a[m]:d[m]}else if("string"===typeof l)for(var u=String(l),O=p(u,n),f=Object.keys(O),x=0,g=f;x<g.length;x++){var w=g[x];a[w]=w in a?O[w]+a[w]:O[w]}}}catch(N){c.e(N)}finally{c.f()}return a};var x=i(31);i(151);var g=i(21);i(152);var w=function(e){var t=e.page,i=void 0===t?1:t,n=e.totalPages,r=void 0===n?1:n,a=e.setPage,s=void 0===a?function(){}:a,c=function(e,t){var i=Math.floor(3.5),n=[];if(t>7)if(n[0]={number:1},n[1]={number:2},n[5]={number:t-1},n[6]={number:t},e<=i){n[5].ellipsis=!0;for(var r=2;r<5;r++)n[r]={number:r+1}}else if(t-e<i){n[1].ellipsis=!0;for(var a=2;a<5;a++)n[a]={number:t-7+a+1}}else{n[1].ellipsis=!0,n[5].ellipsis=!0,n[i]={number:e};for(var s=1;s<2;s++)n[i+s]={number:e+s},n[i-s]={number:e-s}}else for(var c=0;c<t;c++)n[c]={number:c+1,ellipsis:!1};return n.forEach((function(t){t.number===e&&(t.active=!0)})),{pages:n,isLeftArrowEnabled:e>1,isRightArrowEnabled:e<t}}(i,r),o=c.pages,l=c.isLeftArrowEnabled,d=c.isRightArrowEnabled;return Object(m.jsxs)("div",{className:"bp3-button-group pagination",children:[Object(m.jsx)(j.a,{large:!0,disabled:!l,onClick:function(){return s(i-1)},id:"pagination-button-left",children:Object(m.jsx)(j.f,{icon:"chevron-left"})}),o.map((function(e){return Object(m.jsx)(j.a,{large:!0,text:e.ellipsis?"...":e.number,disabled:e.ellipsis,intent:e.active?g.a.PRIMARY:g.a.DEFAULT,onClick:function(){return s(e.number)}},e.number)})),Object(m.jsx)(j.a,{large:!0,disabled:!d,onClick:function(){return s(i+1)},id:"pagination-button-right",children:Object(m.jsx)(j.f,{icon:"chevron-right"})})]})},N={};function y(){return N.port?"".concat(window.location.protocol,"//").concat(window.location.hostname,":").concat(N.port):window.location.origin}var A=function(e){var t=e.error,i=r.a.useState(!1),n=Object(l.a)(i,2),a=n[0],s=n[1],c=t&&t.type;return r.a.useEffect((function(){t&&console.error(t)}),[t,c]),t&&!a&&Object(m.jsxs)("div",{className:"error item-view-error",children:[Object(m.jsx)(j.a,{icon:Object(m.jsx)(j.f,{icon:"cross",color:"white"}),minimal:!0,onClick:function(){return s(!0)}}),"Error [",t.type,"] \u2014 ",JSON.stringify(t.error)]})};var R=function(e){var t=e.itemRenderer,i=void 0===t?u:t,r=e.collectionRenderer,a=void 0===r?O:r,s=e.pagination,c=void 0===s||s,b=e.resultsPerPage,v=void 0===b?12:b,p=Object(n.useState)(c?1:null),f=Object(l.a)(p,2),x=f[0],g=f[1],N=Object(n.useState)(""),R=Object(l.a)(N,2),k=R[0],_=R[1],S=Object(n.useState)(""),E=Object(l.a)(S,2),T=E[0],C=E[1],P=Object(n.useState)(null),F=Object(l.a)(P,2),L=F[0],D=F[1],I=Object(d.useMephistoReview)({page:x,resultsPerPage:v,filters:k,hostname:y()}),M=I.data,$=I.isFinished,B=I.isLoading,J=I.error,U=I.mode,Y=I.totalPages,H=function(e){null!==x&&1!==x&&g(1),_(e)},W=function(){L&&clearTimeout(L),H(T)},G=Object(m.jsx)(j.a,{id:"mephisto-search-button",round:!0,onClick:W,children:"Search"});return"OBO"===U?Object(m.jsx)(o.a,{to:"/".concat(M&&M.id)}):Object(m.jsxs)(m.Fragment,{children:[Object(m.jsx)(j.h,{fixedToTop:!0,children:Object(m.jsxs)("div",{className:"navbar-wrapper",children:[Object(m.jsx)(j.j,{className:"navbar-header",children:Object(m.jsx)(j.k,{children:Object(m.jsx)("b",{children:Object(m.jsx)("pre",{children:"mephisto review"})})})}),Object(m.jsx)(j.j,{align:h.a.CENTER,children:Object(m.jsx)(j.m,{content:"Separate multiple filters with commas",placement:h.a.LEFT,children:Object(m.jsx)(j.g,{id:"mephisto-search",className:"all-item-view-search-bar",leftIcon:"search",onChange:function(e){return t=e.target.value,C(t),L&&clearTimeout(L),void D(setTimeout((function(){H(t)}),3e3));var t},onKeyDown:function(e){"Enter"===e.key&&W()},placeholder:"Filter data...",value:T,rightElement:G})})})]})}),Object(m.jsx)("main",{className:"all-item-view mode-".concat(U),id:"all-item-view-wrapper",children:Object(m.jsxs)("div",{className:"item-dynamic",children:[Object(m.jsx)(A,{error:J}),B?Object(m.jsx)("h1",{className:"all-item-view-message",children:"Loading..."}):$?Object(m.jsx)("h1",{className:"all-item-view-message",children:"Done reviewing! You can close this app now"}):M&&M.length>0?Object(m.jsxs)(m.Fragment,{children:[Object(m.jsx)(a,{items:M,itemRenderer:i}),c&&Y>1?Object(m.jsx)(w,{totalPages:Y,page:x,setPage:g}):null]}):Object(m.jsxs)("div",{className:"all-item-view-message all-item-view-no-data",children:[Object(m.jsxs)("h3",{children:["Thanks for using the ",Object(m.jsx)("code",{children:"$ mephisto review"})," interface. Here are a few ways to get started:"]}),Object(m.jsxs)("h3",{children:["1. Review data from a .csv or"," ",Object(m.jsx)("a",{href:"https://jsonlines.org/",children:".jsonl"})," file"]}),Object(m.jsxs)("pre",{children:["$ cat sample-data",Object(m.jsx)("span",{className:"highlight",children:".json"})," | mephisto review review-app/build/"," ",Object(m.jsx)("span",{className:"highlight",children:"--json"})," --all --stdout"]}),Object(m.jsxs)("pre",{children:["$ cat sample-data",Object(m.jsx)("span",{className:"highlight",children:".csv"})," | mephisto review review-app/build/"," ",Object(m.jsx)("span",{className:"highlight",children:"--csv"})," --all --stdout"]}),Object(m.jsx)("h3",{children:"2. Review data from the Mephisto database"}),Object(m.jsxs)("pre",{children:["$ mephisto review review-app/build/"," ",Object(m.jsx)("span",{className:"highlight",children:"--db mephisto_db_task_name"})," ","--all --stdout"]})]})]})})]})},k=i(25),_=i.n(k),S=i(40),E=i(10),T=j.l.create({className:"recipe-toaster",position:E.a.TOP});var C=function(e){var t=e.itemRenderer,i=void 0===t?u:t,n=e.wrapClass,r=e.allowReview,a=void 0===r||r,s=Object(o.h)().id,l=Object(d.useMephistoReview)({taskId:s,hostname:y()}),b=l.data,O=l.isFinished,v=l.isLoading,p=l.submit,f=l.error,x=l.mode,g=Object(o.g)(),w=function(){"OBO"===x?g.push("/"):T.show({message:"Review response recorded."})},N=function(e){T.show({message:"ERROR: ".concat(e)})},R=f||O||v||null==b,k=!a&&"ALL"===x;return Object(m.jsxs)(m.Fragment,{children:[Object(m.jsx)(j.h,{fixedToTop:!0,children:Object(m.jsxs)("div",{className:"navbar-wrapper",children:[Object(m.jsxs)(j.j,{children:["ALL"===x?Object(m.jsxs)(m.Fragment,{children:[Object(m.jsx)(c.b,{to:"/",style:{textDecoration:"none"},children:Object(m.jsx)(j.a,{intent:"primary",icon:"caret-left",id:"home-button",children:Object(m.jsx)("b",{children:"Mephisto Review"})})}),Object(m.jsx)(j.i,{})]}):null,Object(m.jsx)(j.k,{className:"navbar-header",children:k?Object(m.jsx)("b",{children:"Viewing task:"}):Object(m.jsx)("b",{children:"Please review the following item:"})})]}),k?null:Object(m.jsxs)(j.j,{align:h.a.RIGHT,children:[Object(m.jsx)(j.a,{className:"btn",intent:"danger",disabled:R,id:"reject-button",onClick:Object(S.a)(_.a.mark((function e(){var t;return _.a.wrap((function(e){for(;;)switch(e.prev=e.next){case 0:return e.next=2,p({result:"rejected"});case 2:"SUCCESS"===(t=e.sent)?w():t&&N(t);case 4:case"end":return e.stop()}}),e)}))),children:Object(m.jsx)("b",{children:"REJECT"})}),Object(m.jsx)(j.a,{className:"btn",intent:"success",disabled:R,id:"approve-button",onClick:Object(S.a)(_.a.mark((function e(){var t;return _.a.wrap((function(e){for(;;)switch(e.prev=e.next){case 0:return e.next=2,p({result:"approved"});case 2:"SUCCESS"===(t=e.sent)?w():t&&N(t);case 4:case"end":return e.stop()}}),e)}))),children:Object(m.jsx)("b",{children:"APPROVE"})})]})]})}),Object(m.jsx)("main",{className:"item-view mode-".concat(x),children:v?Object(m.jsxs)("div",{className:"item-dynamic",children:[Object(m.jsx)(A,{error:f}),Object(m.jsx)("h1",{className:"item-view-message",children:"Loading..."})]}):O?Object(m.jsxs)("div",{className:"item-dynamic",children:[Object(m.jsx)(A,{error:f}),Object(m.jsx)("h1",{className:"item-view-message",children:"Done reviewing! You can close this app now"})]}):b?n?Object(m.jsxs)("div",{className:n,children:[Object(m.jsx)(A,{error:f}),Object(m.jsx)(i,{item:b})]}):Object(m.jsxs)(m.Fragment,{children:[Object(m.jsx)(A,{error:f}),Object(m.jsx)(i,{item:b})]}):Object(m.jsx)("div",{className:"item-dynamic",children:Object(m.jsxs)("div",{className:"item-view-message item-view-no-data",children:[Object(m.jsx)(A,{error:f}),Object(m.jsxs)("h3",{children:["Thanks for using the ",Object(m.jsx)("code",{children:"$ mephisto review"})," interface. Here are a few ways to get started:"]}),Object(m.jsxs)("h3",{children:["1. Review data from a .csv or"," ",Object(m.jsx)("a",{href:"https://jsonlines.org/",children:".jsonl"})," file"]}),Object(m.jsxs)("pre",{children:["$ cat sample-data",Object(m.jsx)("span",{className:"highlight",children:".json"})," | mephisto review review-app/build/"," ",Object(m.jsx)("span",{className:"highlight",children:"--json"})," --stdout"]}),Object(m.jsxs)("pre",{children:["$ cat sample-data",Object(m.jsx)("span",{className:"highlight",children:".csv"})," | mephisto review review-app/build/"," ",Object(m.jsx)("span",{className:"highlight",children:"--csv"})," --stdout"]}),Object(m.jsx)("h3",{children:"2. Review data from the Mephisto database"}),Object(m.jsxs)("pre",{children:["$ mephisto review review-app/build/"," ",Object(m.jsx)("span",{className:"highlight",children:"--db mephisto_db_task_name"})," ","--stdout"]})]})})})]})};i(153),i(154),i(155),i(156);function P(e){var t=f(e,["C","the","be","of","from","to","and","a","in","that","have","it","for","not","on","with","by","his","her","up","down"],["id"]);return Object.entries(t).sort((function(e,t){var i=Object(l.a)(e,2),n=(i[0],i[1]),r=Object(l.a)(t,2);r[0];return r[1]-n})).slice(0,10).map((function(e){var t=Object(l.a)(e,2),i=t[0];t[1];return i}))}var F=function(e){var t=e.item.data,i=t.info.payload,n=r.a.useState(!1),a=Object(l.a)(n,2),s=a[0],c=a[1];return Object(m.jsx)("div",{className:"json-item-renderer",children:Object(m.jsxs)(j.c,{elevation:b.a.TWO,interactive:!0,className:"json-item-card",children:[Object(m.jsxs)("p",{style:{fontSize:12},children:[t.uid," \u2014 ",i.length," entries"]}),Object(m.jsx)("img",{role:"presentation",onError:function(e){e.target.onerror=null,c(!0)},src:t.img&&!s?y()+t.img:"data:image/gif;base64,R0lGODlhAQABAIAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw==",alt:"Thumbnail",style:{width:t.img?"100%":"1px"}}),Object(m.jsx)("p",{children:P(i).map((function(e){return Object(m.jsxs)("span",{style:{marginRight:"1em",fontStyle:"italic",display:"inline-block"},children:["#",e]},e)}))})]})})},L=i(67),D=i(66),I=i.n(D);i(175);function M(e){var t,i=e.data,n=y()+(null===i||void 0===i?void 0:i.file),a="narration"===(null===i||void 0===i?void 0:i.taxonomy),s=(null===i||void 0===i||null===(t=i.info)||void 0===t?void 0:t.payload.map((function(e){return function(e,t){if(!t)return e;var i=e.start_time,n=(e.end_time,Object(L.a)(e,["start_time","end_time"]));return Object(x.a)({start_time:i,end_time:i+1},n)}(e,a)})))||[],c=r.a.useState(null),o=Object(l.a)(c,2),d=o[0],h=o[1],b=r.a.useState(null),u=Object(l.a)(b,2),O=u[0],v=u[1],p=r.a.useState(!1),f=Object(l.a)(p,2),w=f[0],N=f[1],A=r.a.useRef(),R=r.a.useState(!1),k=Object(l.a)(R,2),_=k[0],S=k[1],E=s.filter((function(e){return d>=e.start_time-.5&&d<=e.end_time+.5})).map((function(e){return Object(m.jsx)($,{segment:e,duration:O,progress:d,onClick:function(){(null===A||void 0===A?void 0:A.current)&&A.current.seekTo(e.start_time,"seconds"),N(!0)}})}));return Object(m.jsx)("div",{children:Object(m.jsxs)("div",{className:"app-container",children:[Object(m.jsxs)("div",{className:"video-viewer",children:[_?Object(m.jsxs)(j.b,{intent:g.a.WARNING,style:{marginBottom:10},children:["The video was not found. You may not have it downloaded. You can try downloading it with the Ego4D cli:",Object(m.jsxs)("pre",{style:{whiteSpace:"break-spaces"},children:["python -m ego4d.cli.cli --yes --datasets full_scale --output_directory $OUTPUT_DIR --video_uids ",i.uid]})]}):null,Object(m.jsx)(I.a,{url:n,controls:!0,playing:w,ref:A,width:"100%",progressInterval:350,onError:function(e){console.log(e),S(!0)},onProgress:function(e){var t=e.playedSeconds;h(t)},onDuration:v}),Object(m.jsx)("h3",{children:"Active annotations:"}),E.length>0?E:Object(m.jsx)("span",{children:"None"})]}),Object(m.jsxs)("div",{className:"segment-viewer",children:[Object(m.jsx)("h3",{children:"All annotations:"}),s.map((function(e){return Object(m.jsx)($,{useTimePoint:a,segment:e,duration:O,isActive:d>=e.start_time&&d<=e.end_time,onClick:function(){(null===A||void 0===A?void 0:A.current)&&A.current.seekTo(e.start_time,"seconds"),N(!0)}})}))]})]})})}function $(e){var t,i=e.segment,n=e.onClick,r=e.duration,a=e.isActive,s=void 0!==a&&a,c=e.useTimePoint;return Object(m.jsxs)("div",{className:"segment-wrapper "+(s?"active":"inactive"),onClick:n,onKeyDown:n,role:"button",tabIndex:0,children:[c?Object(m.jsx)("div",{className:"duration",children:Object(m.jsxs)("span",{children:[Math.floor(i.start_time/60),":",(i.start_time%60).toFixed(0).padStart(2,"0")]})}):Object(m.jsxs)("div",{className:"duration",children:[Object(m.jsxs)("span",{children:[Math.floor(i.start_time/60),":",(i.start_time%60).toFixed(0).padStart(2,"0")]})," ","\u2014"," ",Object(m.jsxs)("span",{children:[Math.floor(i.end_time/60),":",(i.end_time%60).toFixed(0).padStart(2,"0")]}),"\xa0(",(i.end_time-i.start_time).toFixed(1),"s)"]}),Object(m.jsx)("div",{className:"track",children:r&&Object(m.jsx)("div",{className:"bar",style:{width:100*(i.end_time-i.start_time)/r+"%",marginLeft:100*i.start_time/r+"%"}})}),Object(m.jsx)("div",{className:"segment",children:i.label}),(null===(t=i.tags)||void 0===t?void 0:t.length)>0?Object(m.jsx)("div",{className:"segment",style:{marginTop:10,fontSize:13,fontFamily:"monospace"},children:i.tags.map((function(e,t){return Object(m.jsx)("span",{style:{backgroundColor:"#ccc",border:"1px solid #aaa",borderRadius:3,display:"inline-block",marginRight:5,marginBottom:5,padding:3},children:e||Object(m.jsx)("span",{children:"\xa0"})},t)}))}):null]})}var B=function(e){var t=e.data;return Object(m.jsx)(M,{data:t})};var J=function(e){var t=e.item.data;return t.info.payload,Object(m.jsx)("div",{className:"json-item-renderer",children:Object(m.jsx)(B,{data:t})})};s.a.render(Object(m.jsx)(r.a.StrictMode,{children:Object(m.jsx)(c.a,{children:Object(m.jsxs)(o.d,{children:[Object(m.jsx)(o.b,{path:"/:id",children:Object(m.jsx)(C,{itemRenderer:J,allowReview:!1})}),Object(m.jsx)(o.b,{path:"/",children:Object(m.jsx)(R,{collectionRenderer:O,itemRenderer:F,pagination:!0,resultsPerPage:12})})]})})}),document.getElementById("root"))},57:function(e,t,i){}},[[176,1,2]]]);
//# sourceMappingURL=main.9fedd964.chunk.js.map