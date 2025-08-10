const katex = require('katex');
// const line ='\\begin{array} { r } { { { { R } } _ { 2 } } \\! \\! \\approx \\! \\! \\sum _ { k = 1 } ^ { K } \\! \\gamma \\! \\beta _ { k } \\big | { { \\tilde { X } } } \\big | \\! \\! + \\! \\! \\gamma \\sum _ { k = 1 } ^ { K } \\beta _ { k } \\frac { { { { W } _ { | | k } } } } { \\sqrt { P } { | \\tilde { h } _ { k } | } } \\! \\! + \\! \\! \\gamma \\! \\! \\sqrt { \\! \\! \\sum _ { k = 1 } ^ { K } \\! \\! \\frac { \\beta _ { k } ^ { 2 } } { ( 1 \\! \\! - \\! \\! \\rho _ { k } ) P | \\tilde { h } _ { k } | ^ { 2 } } } { N } . } \\end{array}'
// const line ='{{{{{ a }}  b}}}{ }'
// const line ='{{ a }  b}{ }'
// const line ='{ \\bf p } = { \\frac { \\bf P } { m _ { e } c } }'
// const tree = katex.__parse(line, {strict: 'ignore'})
// console.log(tree[0].body[0].body.body[0].loc)

const fs = require('fs');
const readline = require('readline');

/**
 * 读取文件每行内容，处理后写入新文件
 * @param {string} inputPath - 输入文件路径
 * @param {string} errorPath - 输出文件路径
 */
async function processFile(inputPath, errorPath) {
    // 创建读写流
    const rl = readline.createInterface({
        input: fs.createReadStream(inputPath),
        crlfDelay: Infinity
    });
    const errorStream = fs.createWriteStream(errorPath);

    let lineNumber = 0; // 行号计数器
    let errorNumber = 0;

    try {
        for await (const line of rl) {
            lineNumber++; // 行号从1开始计数
            let tree;
            
            try {
                // 尝试处理当前行
                tree = katex.__parse(line, {strict: 'ignore'})
            } catch (err) {
                // 发生错误时的处理
                // console.error(`${lineNumber}: ${err.message}`);
                errorStream.write(`${lineNumber}: ${err.message}\n`);
                errorNumber++;
            }
            // 写入处理结果（或原始行）
        }
        
        // console.log(`处理完成，共处理 ${lineNumber} 行`);
        console.log(`errors dumped in ${errorPath}`);
        console.log(`parser error lines ${errorNumber}`);
    } catch (err) {
        console.error('文件读取过程中发生错误:', err);
    } finally {
        // 确保写入流关闭
        errorStream.end();
    }
}// 使用示例
const inputFile = './data/dataset/UniMER-1M_merged/train_normalized_.txt';   // 输入文件路径
const errorFile = './data/dataset/UniMER-1M_merged/parse_error.log'; // 输出文件路径

// 执行处理
processFile(inputFile, errorFile).catch(err => console.error('处理过程中发生未捕获错误:', err));
