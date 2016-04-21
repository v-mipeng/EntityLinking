using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Text.RegularExpressions;
using pml.file.reader;
using pml.file.writer;
using pml.type;
using System.Xml;

namespace msra.nlp.el.script.alias.wikipedia
{
    class Script
    {
        public static void ConstructFinalDataset(string source, string des)
        {
            var reader = new LargeFileReader(source);
            var writer = new LargeFileWriter(des, FileMode.Create);
            string line;

            while ((line = reader.ReadLine()) != null)
            {
                var array = line.Split('\t');
                for (var i = 0; i < array.Length;i++)
                {
                    for(var j = 0;j<array.Length && j!=i;j++)
                    {
                        if(array[i].Length == array[j].Length)
                        {
                            writer.WriteLine(array[i] + "\t" + array[j]);
                            writer.WriteLine(array[j] + "\t" + array[i]);
                        }
                        else if (array[i].Length > array[j].Length)
                        {
                            writer.WriteLine(array[i] + "\t" + array[j]);
                        }
                        else
                        {
                            writer.WriteLine(array[j] + "\t" + array[i]);
                        }
                    }
                }
            }
            reader.Close();
            writer.Close();
        }

        public static void ConstructDataFromKbpQuery(string idFile, string queryFile, string des)
        {
            var reader = new LargeFileReader(idFile);
            var writer = new LargeFileWriter(des, FileMode.Create);
            var id2name = new Dictionary<string,string>();
            string line;
            string name;

            while ((line = reader.ReadLine()) != null)
            {
                var array = line.Split('\t');
                id2name[array[0]] = array[1];
            }
            reader.Close();
            XmlDocument doc = new XmlDocument();
            doc.Load(queryFile);
            var nodes = doc.DocumentElement.SelectNodes("/queries/query");
            foreach(XmlNode node in nodes)
            {
                var id = node.LastChild.InnerText;
                if(!id.Equals("NIL") && id2name.TryGetValue(id, out name))
                {
                    var queryName = node.FirstChild.InnerText;
                    if (queryName.Length == name.Length)
                    {
                        writer.WriteLine(queryName + "\t" + name);
                        if (!queryName.Equals(name))
                        {
                            writer.WriteLine(name + "\t" + queryName);
                        }
                    }
                    else if(queryName.Length > name.Length)
                    {
                        writer.WriteLine(queryName + "\t" + name);
                    }
                    else
                    {
                        writer.WriteLine(name + "\t" + queryName);
                    }
                }
            }
            writer.Close();
        }

        public static void AddEntityItself(string source, string des)
        {
            var reader = new LargeFileReader(source);
            var writer = new LargeFileWriter(des, FileMode.Create);
            var dic = new Dictionary<string, HashSet<string>>();
            string line;

            while ((line = reader.ReadLine()) != null)
            {
                var array = line.Split('\t');
                try
                {
                    var set = dic[array[0]];
                    set.Add(array[1]);
                }
               catch (Exception)
               {
                   var set = new HashSet<string>();
                   set.Add(array[1]);
                   dic[array[0]] = set;
               }
            }
            reader.Close();
            foreach(var item in dic)
            {
                writer.WriteLine(item.Key + "\t" + item.Key);
                foreach(var word in item.Value)
                {
                    writer.WriteLine(item.Key + "\t" + word);
                }
            }
            writer.Close();
        }

        public static void AdjustInputOutputOrder(string source, string des)
        {
            var reader = new LargeFileReader(source);
            var writer = new LargeFileWriter(des, FileMode.Create);
            string line;

            while ((line = reader.ReadLine()) != null)
            {
                var array = line.Split('\t');
                if(array[0].Length<array[1].Length)
                {
                    writer.WriteLine(array[1] + '\t' + array[0]);
                }
                else
                {
                    writer.WriteLine(line);
                }
            }
            reader.Close();
            writer.Close();
        }
    }
}
