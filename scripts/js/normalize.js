const katex = require('katex');

function removeNestedBraces(line) {
    // 用于存储所有替换记录的数组
    const replacementLogs = [];
    
    // 递归处理函数
    function processNode(currentNode) {
        // 如果是数组，递归处理每个元素
        if (Array.isArray(currentNode)) {
            return currentNode.map(item => processNode(item));
        }
        
        // 如果不是对象，直接返回
        if (typeof currentNode !== 'object' || currentNode === null) {
            return currentNode;
        }
        
        // 先处理body中的所有子节点
        if (currentNode.body) {
            currentNode.body = processNode(currentNode.body);
            
            // 检查是否需要替换当前节点
            if (currentNode.type === 'ordgroup' && 
                Array.isArray(currentNode.body) && 
                currentNode.body.length === 1 && 
                currentNode.body[0] && 
                currentNode.body[0].type === 'ordgroup') {
                
                // 记录替换信息（只在有loc值时记录）
                if (currentNode.loc && currentNode.body[0].loc) {
                    replacementLogs.push({
                        replacedSpan: [currentNode.loc.start, currentNode.loc.end],    // 被取代节点的loc
                        replacingSpan: [currentNode.body[0].loc.start, currentNode.body[0].loc.end]  // 取代节点的loc
                    });
                }
                
                // 用子节点替换当前节点，并继续处理替换后的节点
                return processNode(currentNode.body[0]);
            }
        }
        
        // 处理其他可能包含子节点的属性
        for (const key in currentNode) {
            if (key !== 'body' && typeof currentNode[key] === 'object' && currentNode[key] !== null) {
                currentNode[key] = processNode(currentNode[key]);
            }
        }
        
        return currentNode;
    }
    
    // 处理根节点
    const tree = katex.__parse(line, {strict: 'ignore'})
    const modifiedTree = processNode(tree);
    
    const mask = new Array(line.length).fill(true)
    // 返回处理后的树和替换记录
    for (const log of replacementLogs){
        if (log.replacedSpan[0] <= log.replacingSpan[0] && log.replacedSpan[1] >= log.replacingSpan[1]){
            for (let i = log.replacedSpan[0]; i < log.replacingSpan[0]; i++){
                mask[i] = false
            }
            for (let i = log.replacingSpan[1]; i < log.replacedSpan[1]; i++){
                mask[i] = false
            }
        }
    }
    const charArray = line.split('')
    const filteredArray = charArray.filter((_, index) => mask[index])
    return filteredArray.join('')
}

// const res = removeNestedBraces(line)

const fs = require('fs');
const readline = require('readline');
const { error } = require('console');

/**
 * 读取文件每行内容，处理后写入新文件
 * @param {string} inputPath - 输入文件路径
 * @param {string} outputPath - 输出文件路径
 */
async function processFile(inputPath, outputPath, errorPath) {
    // 创建读写流
    const rl = readline.createInterface({
        input: fs.createReadStream(inputPath),
        crlfDelay: Infinity
    });
    console.log("normalize "+ inputPath)
    const writeStream = fs.createWriteStream(outputPath);
    const errorStream = fs.createWriteStream(errorPath);

    let lineNumber = 0; // 行号计数器
    let errorNumber = 0;

    try {
        for await (const line of rl) {
            lineNumber++; // 行号从1开始计数
            let result;
            
            try {
                // 尝试处理当前行
                result = removeNestedBraces(line);
            } catch (err) {
                // 发生错误时的处理
                // console.error(`${lineNumber}: ${err.message}`);
                errorStream.write(`${lineNumber}: ${err.message}\n`);
                result = line; // 使用未经处理的原始行
                errorNumber++;
            }
            
            // 写入处理结果（或原始行）
            writeStream.write(result + '\n');
        }
        
        // console.log(`处理完成，共处理 ${lineNumber} 行`);
        console.log(`saved to ${outputPath}`);
        console.log(`errors dumped to ${errorPath}`);
    } catch (err) {
        console.error('文件读取过程中发生错误:', err);
    } finally {
        // 确保写入流关闭
        writeStream.end();
    }

}// 使用示例
const inputFile = './data/dataset/UniMER-1M_merged/train_normalized.txt';   // 输入文件路径
const outputFile = './data/dataset/UniMER-1M_merged/train_normalized_.txt'; // 输出文件路径
const errorFile = './data/dataset/UniMER-1M_merged/error.log'; // 输出文件路径

// 执行处理
processFile(inputFile, outputFile, errorFile).catch(err => console.error('处理过程中发生未捕获错误:', err));
