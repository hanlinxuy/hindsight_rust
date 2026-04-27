/// End-to-end profile building test: retain 50 facts → recall → reflect a user profile.
/// Uses DeepSeek for LLM + homelinux for embeddings.
use hindsight_core::engine::MemoryEngine;
use hindsight_core::models::Disposition;
use std::env;

fn create_engine() -> Option<MemoryEngine> {
    let api_key = env::var("HINDSIGHT_TEST_API_KEY").ok()?;
    if api_key.is_empty() { return None; }
    let base_url = env::var("HINDSIGHT_TEST_BASE_URL")
        .unwrap_or_else(|_| "https://api.openai.com".into());
    let model = env::var("HINDSIGHT_TEST_MODEL")
        .unwrap_or_else(|_| "gpt-4o-mini".into());
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("profile_test.db");
    std::mem::forget(dir);

    let mut engine = MemoryEngine::init(db_path.to_str().unwrap(), &api_key, &base_url, &model).ok()?;
    if let Ok(emb_url) = env::var("HINDSIGHT_TEST_EMBEDDING_BASE_URL") {
        let emb_model = env::var("HINDSIGHT_TEST_EMBEDDING_MODEL")
            .unwrap_or_else(|_| "bge-large-zh-v1.5".into());
        engine = engine.with_embedding(&emb_model, &emb_url);
    }
    Some(engine)
}

macro_rules! maybe_skip {
    () => {
        if create_engine().is_none() {
            eprintln!("Skipping - set HINDSIGHT_TEST_API_KEY to run");
            return;
        }
    };
}

/// 50 facts about a fictional user "张伟" (Zhang Wei)
const FACTS: &[&str] = &[
    // Batch 1: Basic identity & background
    "张伟今年28岁，住在上海浦东新区，是一名全栈工程师。",
    "张伟毕业于浙江大学计算机科学专业，2020年本科毕业。",
    "张伟身高175cm，平时戴黑框眼镜，留着短发。",
    "张伟有一个姐姐叫张丽，姐姐在杭州做产品经理。",
    "张伟的父母退休前都是中学教师，住在老家温州。",

    // Batch 2: Work & career
    "张伟目前在一家AI创业公司「星云智能」担任技术负责人。",
    "张伟的团队有12个人，他同时管理前端和后端两个小组。",
    "张伟擅长Rust、Python和TypeScript三种编程语言。",
    "张伟每天都在用VS Code写代码，主题是One Dark Pro。",
    "张伟公司用的是飞书做协作，他最常用的快捷键是Cmd+P。",

    // Batch 3: Technical preferences
    "张伟偏好函数式编程风格，喜欢用Rust的trait系统设计抽象。",
    "张伟认为TypeScript的类型系统比Python好，但Python做原型更快。",
    "张伟在GitHub上有超过200个star的仓库，主要是Rust工具库。",
    "张伟习惯用git rebase而不是merge来保持提交历史整洁。",
    "张伟的技术博客「锈迹斑斑」每月有5000左右访问量，主要写Rust教程。",

    // Batch 4: Daily habits
    "张伟每天早上7点起床，先喝一杯美式咖啡再开始工作。",
    "张伟通勤骑共享单车15分钟到公司，不喜欢挤地铁。",
    "张伟午饭通常在公司楼下食堂解决，最常点的是红烧牛肉面。",
    "张伟下午3点左右会吃一把坚果当零食，尤其喜欢吃腰果。",
    "张伟晚上一般加班到9点，回家后还要写一小时技术博客。",

    // Batch 5: Hobbies & interests
    "张伟周末喜欢打羽毛球，在浦东一个业余俱乐部打球三年了。",
    "张伟是利物浦足球俱乐部的铁杆球迷，2018年开始关注英超。",
    "张伟有一个Steam账号，最近在玩博德之门3，已经通关两次。",
    "张伟喜欢看科幻小说，最喜欢的作者是刘慈欣和特德·姜。",
    "张伟在学日语，目前已经过了N3水平考试，目标是一年内过N2。",

    // Batch 6: Social & relationships
    "张伟有一个女朋友叫林小雨，是公司的设计师，两人交往两年了。",
    "张伟最好的朋友叫王磊，大学室友，现在在字节跳动做后端。",
    "张伟每个月和大学同学聚会一次，通常在徐汇区的日料店「鱼心」。",
    "张伟不太喜欢大型社交活动，但会参加Rust Shanghai的线下meetup。",
    "张伟在微信上有三个活跃群：大学群、Rust中文社区群、羽毛球俱乐部群。",

    // Batch 7: Personality traits
    "张伟是个内向的人，但聊到技术话题时会变得很健谈。",
    "张伟做事很注重细节，代码审查时会逐行检查逻辑。",
    "张伟不喜欢加班文化，但经常为了解决有趣的技术问题自愿加班。",
    "张伟有一个习惯：每次解决完bug会奖励自己一杯奶茶。",
    "张伟的MBTI是INTJ，他认同这个分类。",

    // Batch 8: Health & lifestyle
    "张伟每周去健身房两次，主要练力量训练和核心。",
    "张伟有点轻度近视，左眼300度右眼275度。",
    "张伟去年体检发现尿酸偏高，医生建议少吃海鲜和啤酒。",
    "张伟睡眠质量不太好，经常失眠，在尝试冥想改善。",
    "张伟不用Apple产品，用的是小米手机和ThinkPad笔记本。",

    // Batch 9: Financial & goals
    "张伟月薪税后28K，每月能存下1万左右。",
    "张伟在攒钱买房，目标是浦东唐镇的一套两居室。",
    "张伟投资了一些指数基金，主要是沪深300和中证500。",
    "张伟的职业目标是五年内成为CTO或开始自己的创业项目。",
    "张伟正在考虑考一个AWS Solutions Architect认证。",

    // Batch 10: Recent events
    "张伟上周参加了RustConf 2024上海的线上直播，听了关于嵌入式Rust的分享。",
    "张伟最近在帮女朋友林小雨设计一个个人作品集网站，用的Next.js。",
    "张伟上个月回温州看望父母，给妈妈买了一条丝巾当生日礼物。",
    "张伟的公司刚拿到A轮融资2000万，他作为技术负责人持有2%的期权。",
    "张伟计划今年国庆节和林小雨去日本旅行，想去京都看枫叶。",
];

#[tokio::test]
async fn test_build_user_profile() {
    maybe_skip!();
    let engine = create_engine().unwrap();

    // 1. Create a bank for this user
    let bank = engine.create_bank("zhangwei_profile", Some(Disposition {
        skepticism: 3, literalism: 2, empathy: 4,
    })).unwrap();
    let bank_id = &bank.id;
    println!("Created bank: {}", bank_id);

    // 2. Retain all 50 facts in batches of 5 (10 batches)
    let mut total_facts = 0usize;
    let mut total_entities = 0usize;
    let batch_size = 5;

    for (batch_idx, chunk) in FACTS.chunks(batch_size).enumerate() {
        let content = chunk.join(" ");
        let result = engine.retain(bank_id, &content).await.unwrap();
        total_facts += result.facts_extracted;
        total_entities += result.entities_extracted;
        println!(
            "Batch {}: {} facts extracted, {} entities, doc_id={}",
            batch_idx + 1, result.facts_extracted, result.entities_extracted, result.document_id
        );
    }

    println!("\n=== Retain Summary ===");
    println!("Total facts extracted: {}", total_facts);
    println!("Total entities extracted: {}", total_entities);

    // Verify stored memories
    let all_memories = engine.list_memories(bank_id, 200, 0).unwrap();
    println!("Total memory units in bank: {}", all_memories.len());
    assert!(all_memories.len() >= 30, "Should have at least 30 memory units from 50 facts, got {}", all_memories.len());

    // 3. Recall queries - test different retrieval angles
    println!("\n=== Recall Queries ===");

    let queries = vec![
        "张伟的技术背景和编程能力",
        "张伟的日常生活习惯",
        "张伟的社交关系和家人",
        "张伟的健康状况",
        "张伟的职业目标和财务状况",
        "张伟最近的计划和活动",
    ];

    for query in &queries {
        let results = engine.recall(bank_id, query, 10).await.unwrap();
        println!("\nQuery: '{}'", query);
        println!("  Results: {}", results.len());
        for r in &results {
            println!("    [{:.3}] {}", r.score, r.unit.content.chars().take(60).collect::<String>());
        }
    }

    // 4. Reflect: Build a comprehensive user profile
    println!("\n=== Reflect: User Profile ===");

    let profile_queries = vec![
        "请根据所有记忆，总结张伟的个人画像，包括性格特点、技术能力、生活方式、社交关系和未来规划。",
        "张伟作为一个程序员的核心技术价值观是什么？他偏好的技术栈和工作方式有哪些特征？",
        "从记忆中分析张伟的情感状态和生活满意度，他有哪些压力来源和应对方式？",
    ];

    for query in &profile_queries {
        let result = engine.reflect(bank_id, query).await.unwrap();
        println!("\nReflect Query: {}", query);
        println!("Memories used: {}", result.memories_used);
        println!("Answer:\n{}\n", result.answer);
        println!("{}", "-".repeat(80));
    }

    // 5. Try consolidation to find patterns
    println!("\n=== Consolidation ===");
    let consolidate_result = engine.consolidate(bank_id, 0.6, 5).await.unwrap();
    println!("Groups found: {}", consolidate_result.groups_found);
    println!("Mental models created: {}", consolidate_result.mental_models_created);

    // Check for mental models
    let all_after = engine.list_memories(bank_id, 200, 0).unwrap();
    let mental_models: Vec<_> = all_after.iter().filter(|m| format!("{:?}", m.fact_type).contains("MentalModel")).collect();
    if !mental_models.is_empty() {
        println!("\nMental Models Created:");
        for mm in &mental_models {
            println!("  [{}] {} (confidence: {:.2})",
                mm.id.chars().take(8).collect::<String>(),
                mm.content.chars().take(100).collect::<String>(),
                mm.confidence
            );
        }
    }

    // 6. Test inject_observation: directly add facts without LLM
    println!("\n=== Observation Injection ===");
    let inject_facts = vec![
        "张伟今天感冒了，请了半天病假。",
        "张伟刚刚通过了AWS Solutions Architect Associate考试。",
        "张伟和林小雨计划下个月搬到一起住。",
    ];
    for fact in &inject_facts {
        let id = engine.inject_observation(bank_id, fact, None, vec!["observation".into()], None).await.unwrap();
        println!("Injected: {} -> {}", id, fact);
    }
    let after_inject = engine.list_memories(bank_id, 200, 0).unwrap();
    println!("Total memories after injection: {}", after_inject.len());

    // 7. Test graph query
    println!("\n=== Graph Query ===");
    if let Some(first) = all_memories.first() {
        let graph = engine.query_graph(&first.id, 2, vec![]).unwrap();
        println!("Graph from {}: {} reachable nodes (max 2 hops)", first.id.chars().take(8).collect::<String>(), graph.len());
        for (unit, hops) in graph.iter().take(10) {
            println!("  [hop={}] {}", hops, unit.content.chars().take(60).collect::<String>());
        }
    }

    // 8. Test get_related
    println!("\n=== Entity Related Memories ===");
    let related = engine.get_related(bank_id, "张伟").unwrap();
    println!("Memories related to entity '张伟': {}", related.len());
    for m in related.iter().take(5) {
        println!("  [{}] {}", m.fact_type.as_ref(), m.content.chars().take(60).collect::<String>());
    }
}
